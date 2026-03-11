// Package inference provides logic to fetch and prepare model artifacts for inference images.
package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	orasImage         = "ghcr.io/oras-project/oras:v1.2.0"
	ollamaRegistryURL = "registry.ollama.ai"
)

// handleOCI handles OCI artifact downloading and processing.
func handleOCI(source string, s llb.State, platform specs.Platform) llb.State {
	toolingImage := llb.Image(orasImage, llb.Platform(platform))

	artifactURL := strings.TrimPrefix(source, "oci://")
	var script string

	if strings.HasPrefix(artifactURL, ollamaRegistryURL) {
		// Reuse existing specialized logic
		modelName, orasCmd := handleOllamaRegistry(artifactURL)
		script = fmt.Sprintf("apk add --no-cache jq curl && %s", orasCmd)
		toolingImage = toolingImage.Run(utils.Sh(script)).Root()
		modelPath := fmt.Sprintf("/models/%s", modelName)
		s = s.File(
			llb.Copy(toolingImage, modelName, modelPath, createCopyOptions()...),
			llb.WithCustomName("Copying "+artifactURL+" to "+modelPath),
		)
		return s
	}

	// Generic (ModelPack) selects the first application/vnd.cncf.model.weight.* layer.
	orasCmd := handleGenericModelPack(artifactURL)
	script = fmt.Sprintf("apk add --no-cache jq curl && %s", orasCmd)
	toolingImage = toolingImage.Run(utils.Sh(script)).Root()
	// Copy all files from /download to /models
	s = s.File(
		llb.Copy(toolingImage, "/download/", "/models/", &llb.CopyInfo{
			CopyDirContentsOnly: true,
			CreateDestPath:      true,
		}),
		llb.WithCustomName("Copying weight layer from "+artifactURL+" to /models/"),
	)
	return s
}

// handleOllamaRegistry handles the Ollama registry specific download.
func handleOllamaRegistry(artifactURL string) (string, string) {
	artifactURLWithoutTag := strings.Split(artifactURL, ":")[0]
	tag := strings.Split(artifactURL, ":")[1]
	modelName := strings.Split(artifactURLWithoutTag, "/")[2]
	orasCmd := fmt.Sprintf("oras blob fetch %[1]s@$(curl https://%[2]s/v2/library/%[3]s/manifests/%[4]s | jq -r '.layers[] | select(.mediaType == \"application/vnd.ollama.image.model\").digest') --output %[3]s", artifactURLWithoutTag, ollamaRegistryURL, modelName, tag)
	return modelName, orasCmd
}

// handleGenericModelPack builds an oras command that pulls the artifact,
// automatically using org.opencontainers.image.title for filenames.
// For localhost registries (localhost:* or 127.0.0.1:*), uses --insecure flag with a warning.
func handleGenericModelPack(artifactURL string) string {
	// Determine if this is a localhost registry that may need insecure flag
	isLocalhost := strings.HasPrefix(artifactURL, "localhost:") ||
		strings.HasPrefix(artifactURL, "127.0.0.1:") ||
		strings.HasPrefix(artifactURL, "::1:")

	insecureFlag := ""
	warningMsg := ""
	if isLocalhost {
		insecureFlag = "--insecure"
		warningMsg = "echo '[WARNING] Using insecure connection for localhost registry' >&2\n"
	}

	cmd := fmt.Sprintf(`set -e
ref=%[1]s
%[2]s
mkdir -p /download
cd /download
echo "Pulling artifact from $ref" >&2
if ! oras pull %[3]s "$ref" 2>/tmp/oras-error.log; then
	echo "Failed to pull artifact from $ref" >&2
	cat /tmp/oras-error.log >&2
	exit 1
fi
echo "Downloaded files:" >&2
ls -lh /download
`, artifactURL, warningMsg, insecureFlag)

	return cmd
}

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

// ParseHuggingFaceURL converts a huggingface:// URL to https:// URL with optional branch support.
func ParseHuggingFaceURL(source string) (string, string, error) {
	baseURL := "https://huggingface.co/"
	modelPath := strings.TrimPrefix(source, "huggingface://")

	// Split the model path to check for branch specification
	parts := strings.Split(modelPath, "/")

	if len(parts) < 3 {
		return "", "", errors.New("invalid Hugging Face URL format")
	}

	namespace := parts[0]
	model := parts[1]
	var branch, modelFile string

	if len(parts) == 4 {
		// URL includes branch: "huggingface://{namespace}/{model}/{branch}/{file}"
		branch = parts[2]
		modelFile = parts[3]
	} else {
		// URL does not include branch, default to main: "huggingface://{namespace}/{model}/{file}"
		branch = "main"
		modelFile = parts[2]
	}

	// Construct the full URL
	fullURL := fmt.Sprintf("%s%s/%s/resolve/%s/%s", baseURL, namespace, model, branch, modelFile)
	return fullURL, modelFile, nil
}

// handleHuggingFace handles Hugging Face model downloads with branch support.
// Supports both single-file downloads (huggingface://namespace/model/file) and
// full repo downloads (huggingface://namespace/model) by enumerating repo files
// via the HuggingFace API and downloading each with BuildKit's llb.HTTP.
func handleHuggingFace(source string, s llb.State) (llb.State, error) {
	// Try single-file download first (3+ parts)
	hfURL, modelName, err := ParseHuggingFaceURL(source)
	if err == nil {
		opts := []llb.HTTPOption{llb.Filename(modelName)}
		m := llb.HTTP(hfURL, opts...)
		modelPath := fmt.Sprintf("/models/%s", modelName)
		s = s.File(
			llb.Copy(m, modelName, modelPath, createCopyOptions()...),
			llb.WithCustomName("Copying "+modelName+" from Hugging Face to "+modelPath),
		)
		return s, nil
	}

	// Fall back to full repo download (2 parts: namespace/model)
	spec, err := ParseHuggingFaceSpec(source)
	if err != nil {
		return llb.State{}, fmt.Errorf("invalid Hugging Face URL format: %w", err)
	}

	files, err := listHuggingFaceRepoFiles(spec.Namespace, spec.Model, spec.Revision)
	if err != nil {
		return llb.State{}, fmt.Errorf("listing HuggingFace repo files: %w", err)
	}

	modelDir := fmt.Sprintf("/models/%s/%s", spec.Namespace, spec.Model)
	for _, file := range files {
		fileURL := fmt.Sprintf("https://huggingface.co/%s/%s/resolve/%s/%s", spec.Namespace, spec.Model, spec.Revision, file)
		fileName := path.Base(file)
		opts := []llb.HTTPOption{llb.Filename(fileName)}
		m := llb.HTTP(fileURL, opts...)

		destPath := fmt.Sprintf("%s/%s", modelDir, file)
		s = s.File(
			llb.Copy(m, fileName, destPath, createCopyOptions()...),
			llb.WithCustomNamef("Copying %s from HuggingFace to %s", file, destPath),
		)
	}
	return s, nil
}

// listHuggingFaceRepoFiles returns the list of files in a HuggingFace repo,
// excluding non-essential files like .gitattributes and README.md.
func listHuggingFaceRepoFiles(namespace, model, revision string) ([]string, error) {
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/%s/revision/%s", namespace, model, revision)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(apiURL) //nolint:gosec
	if err != nil {
		return nil, fmt.Errorf("fetching repo info: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HuggingFace API returned status %d for %s/%s", resp.StatusCode, namespace, model)
	}

	var result struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding HuggingFace API response: %w", err)
	}

	skipFiles := map[string]bool{
		".gitattributes": true,
		"README.md":      true,
		"LICENSE":        true,
	}

	var files []string
	for _, s := range result.Siblings {
		if !skipFiles[s.RFilename] {
			files = append(files, s.RFilename)
		}
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no downloadable files found in %s/%s", namespace, model)
	}
	return files, nil
}

// handleLocal handles copying from local paths.
func handleLocal(source string, s llb.State) llb.State {
	s = s.File(
		llb.Copy(llb.Local("context"), source, "/models/", createCopyOptions()...),
		llb.WithCustomName("Copying "+utils.FileNameFromURL(source)+" to /models"),
	)
	return s
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

// HuggingFaceSpec represents a parsed huggingface:// reference.
// Supported forms:
//
//	huggingface://namespace/model                -> revision: main
//	huggingface://namespace/model@rev            -> explicit revision
//	huggingface://namespace/model:rev            -> (legacy separator) explicit revision
//	huggingface://namespace/model@rev/path/to    -> with subpath (ignored by current callers)
//	huggingface://namespace/model/path/to        -> implicit main revision with subpath
//
// For current usage we only need Namespace, Model, Revision; subpath is ignored.
type HuggingFaceSpec struct {
	Namespace string
	Model     string
	Revision  string
	SubPath   string // optional; empty means whole repo
}

var hfSpecPattern = regexp.MustCompile(`^huggingface://([^/]+)/([^/@:]+)(?:[@:]([^/]+))?(?:/(.*))?$`)

// ParseHuggingFaceSpec parses a huggingface:// reference into its components.
// Defaults revision to "main" when omitted.
func ParseHuggingFaceSpec(src string) (*HuggingFaceSpec, error) {
	if !strings.HasPrefix(src, "huggingface://") {
		return nil, fmt.Errorf("not a huggingface source: %s", src)
	}
	m := hfSpecPattern.FindStringSubmatch(src)
	if m == nil {
		return nil, fmt.Errorf("invalid huggingface spec: %s", src)
	}
	spec := &HuggingFaceSpec{Namespace: m[1], Model: m[2], Revision: "main"}
	if m[3] != "" {
		spec.Revision = m[3]
	}
	if m[4] != "" {
		spec.SubPath = m[4]
	}
	// Basic validation: no empty pieces
	if spec.Namespace == "" || spec.Model == "" {
		return nil, errors.New("namespace and model required")
	}
	return spec, nil
}
