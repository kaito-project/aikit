package inference

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strings"
	"time"

	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	digest "github.com/opencontainers/go-digest"
)

// HFSpec captures a parsed huggingface:// reference.
type HFSpec struct {
	Namespace string
	Model     string
	Revision  string
	File      string // empty if whole repo
	WholeRepo bool
}

func ParseHuggingFaceSpec(source string) (HFSpec, error) {
	var spec HFSpec
	modelPath := strings.TrimPrefix(source, "huggingface://")
	parts := strings.Split(modelPath, "/")
	if len(parts) < 2 { return spec, errors.New("invalid Hugging Face URL format") }
	spec.Namespace = parts[0]
	spec.Model = parts[1]
	remain := parts[2:]
	if len(remain) == 0 { spec.Revision = "main"; spec.WholeRepo = true; return spec, nil }
	if len(remain) == 1 { seg := remain[0]; if strings.Contains(seg, ".") { spec.Revision = "main"; spec.File = seg; return spec, nil }; spec.Revision = seg; spec.WholeRepo = true; return spec, nil }
	spec.Revision = remain[0]
	spec.File = strings.Join(remain[1:], "/")
	return spec, nil
}

func ListHuggingFaceFiles(spec HFSpec) ([]string, error) { return ListHuggingFaceFilesWithCA(spec, "") }

// ListHuggingFaceFilesWithCA allows supplying additional PEM encoded CA certificates (concatenated) to validate TLS.
// If caPEM is empty, system roots are used. This replaces prior insecure skip mode.
func ListHuggingFaceFilesWithCA(spec HFSpec, caPEM string) ([]string, error) {
	if !spec.WholeRepo { return []string{spec.File}, nil }
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/%s/tree/%s?recursive=1", spec.Namespace, spec.Model, spec.Revision)
	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil { return nil, err }
	req.Header.Set("User-Agent", "aikit-buildkit")
	req.Header.Set("Accept", "application/json")
	// Build transport with optional extra CAs.
	var tlsConfig *tls.Config
	if caPEM != "" {
		pool, err := x509.SystemCertPool()
		if err != nil || pool == nil { pool = x509.NewCertPool() }
		if ok := pool.AppendCertsFromPEM([]byte(caPEM)); !ok {
			return nil, errors.New("failed to append provided CA certificate(s)")
		}
		tlsConfig = &tls.Config{RootCAs: pool}
	}
	tr := &http.Transport{TLSClientConfig: tlsConfig}
	client := &http.Client{Timeout: 30 * time.Second, Transport: tr}
	resp, err := client.Do(req)
	if err != nil { return nil, err }
	defer resp.Body.Close()
	if resp.StatusCode != 200 { return nil, fmt.Errorf("huggingface api status %d", resp.StatusCode) }
	b, err := io.ReadAll(resp.Body); if err != nil { return nil, err }
	var entries []struct { Path string `json:"path"`; Type string `json:"type"` }
	if err := json.Unmarshal(b, &entries); err != nil { return nil, err }
	out := []string{}
	for _, e := range entries {
		// Some responses may omit Type; treat missing as file if path looks like a leaf.
		if e.Type == "file" || e.Type == "" {
			if e.Path != "" && !strings.HasSuffix(e.Path, "/") {
				out = append(out, e.Path)
			}
		}
	}
	if len(out) == 0 {
		return nil, errors.New("huggingface list returned zero files (possible API shape change or permission issue)")
	}
	return out, nil
}

// Legacy helper retained for prior callers that only support single file.
// Legacy single-file helper retained for old callers elsewhere.
func ParseHuggingFaceURL(source string) (string, string, error) { //nolint:revive
	spec, err := ParseHuggingFaceSpec(source); if err != nil { return "", "", err }
	if spec.WholeRepo { return "", "", errors.New("whole-repo reference not supported in this code path") }
	fullURL := fmt.Sprintf("https://huggingface.co/%s/%s/resolve/%s/%s?download=1", spec.Namespace, spec.Model, spec.Revision, spec.File)
	return fullURL, path.Base(spec.File), nil
}

// handleOllamaRegistry handles the Ollama registry specific download.
// handleOllamaRegistry removed (unused after refactor)

// handleHTTP handles HTTP(S) downloads.
func handleHTTP(source, name, sha256 string, s llb.State) llb.State {
	opts := []llb.HTTPOption{llb.Filename(utils.FileNameFromURL(source))}
	if sha256 != "" {
		digest := digest.NewDigestFromEncoded(digest.SHA256, sha256)
		opts = append(opts, llb.Checksum(digest))
	}

	m := llb.HTTP(source, opts...)
	modelPath := "/models/" + utils.FileNameFromURL(source)
	if strings.Contains(name, "/") {
		modelPath = "/models/" + path.Dir(name) + "/" + utils.FileNameFromURL(source)
	}

	s = s.File(
		llb.Copy(m, utils.FileNameFromURL(source), modelPath, createCopyOptions()...),
		llb.WithCustomName("Copying "+utils.FileNameFromURL(source)+" to "+modelPath),
	)
	return s
}

// parseHuggingFaceURL converts a huggingface:// URL to https:// URL with optional branch support.
// (Removed duplicate legacy ParseHuggingFaceURL definition)

// handleHuggingFace handles Hugging Face model downloads with branch support.
func handleHuggingFace(source string, s llb.State) (llb.State, error) {
	spec, err := ParseHuggingFaceSpec(source); if err != nil { return llb.State{}, err }
	files, err := ListHuggingFaceFiles(spec); if err != nil { return llb.State{}, err }
	for _, f := range files {
		url := fmt.Sprintf("https://huggingface.co/%s/%s/resolve/%s/%s?download=1", spec.Namespace, spec.Model, spec.Revision, f)
		opts := []llb.HTTPOption{ llb.Filename(path.Base(f)) }
		m := llb.HTTP(url, opts...)
		target := "/models/" + f
		s = s.File(
			llb.Copy(m, path.Base(f), target, createCopyOptions()...),
			llb.WithCustomName("Copying "+f+" from Hugging Face"),
		)
	}
	return s, nil
}

// handleLocal handles copying from local paths.
func handleLocal(source string, s llb.State) llb.State {
	s = s.File(
		llb.Copy(llb.Local("context"), source, "/models/", createCopyOptions()...),
		llb.WithCustomName("Copying "+utils.FileNameFromURL(source)+" to /models"),
	)
	return s
}

// handleOCI placeholder: previously implemented elsewhere; minimal no-op copy until restored.
func handleOCI(source string, s llb.State, platform interface{}) llb.State { //nolint:revive
	// TODO: Re-implement OCI artifact fetching if needed.
	return s
}

// extractModelName extracts the model name from an OCI artifact URL.
func extractModelName(artifactURL string) string {
	modelName := path.Base(artifactURL)
	modelName = strings.Split(modelName, ":")[0]
	modelName = strings.Split(modelName, "@")[0]
	return modelName
}

// createCopyOptions returns the common llb.CopyOption used in file operations.
func createCopyOptions() []llb.CopyOption {
	mode := llb.ChmodOpt{
		Mode: os.FileMode(0o444),
	}
	return []llb.CopyOption{
		&llb.CopyInfo{
			CreateDestPath: true,
			Mode:           &mode,
		},
	}
}
