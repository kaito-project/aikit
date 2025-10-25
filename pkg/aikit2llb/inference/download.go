// Package inference provides logic to fetch and prepare model artifacts for inference images.
package inference

import (
	"errors"
	"fmt"
	"os"
	"path"
	"regexp"
	"strings"

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
	modelName, orasCmd := handleGenericModelPack(artifactURL)
	script = fmt.Sprintf("apk add --no-cache jq curl && %s", orasCmd)
	toolingImage = toolingImage.Run(utils.Sh(script)).Root()
	modelPath := fmt.Sprintf("/models/%s", modelName)
	s = s.File(
		llb.Copy(toolingImage, modelName, modelPath, createCopyOptions()...),
		llb.WithCustomName("Copying weight layer from "+artifactURL+" to "+modelPath),
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

// handleGenericModelPack builds an oras command that:
// 1. Fetches the manifest from the registry
// 2. Extracts the first layer whose mediaType starts with application/vnd.cncf.model.weight.
// 3. Downloads that blob to a file named after the model (base ref name) OR annotation title if present.
// For localhost registries (localhost:* or 127.0.0.1:*), uses --insecure flag with a warning.
func handleGenericModelPack(artifactURL string) (string, string) {
	modelName := extractModelName(artifactURL)

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
tmp=/tmp/manifest.json
%[3]s
# Fetch manifest
if ! oras manifest fetch "$ref" -o "$tmp" %[4]s 2>/tmp/oras-error.log; then
	echo "Failed to fetch manifest from $ref" >&2
	cat /tmp/oras-error.log >&2
	exit 1
fi
layerDigest=$(jq -r '.layers[] | select(.mediaType | startswith("application/vnd.cncf.model.weight.")) | .digest' "$tmp" | head -n1)
if [ -z "$layerDigest" ]; then
	echo "Error: No application/vnd.cncf.model.weight.* layer found in manifest. Verify that the artifact was packaged with the modelpack target." >&2
	echo "Available layers:" >&2
	jq -r '.layers[] | "\(.mediaType): \(.digest)"' "$tmp" >&2
	exit 1
fi
title=$(jq -r '.layers[] | select(.digest=="'$layerDigest'") | .annotations["org.opencontainers.image.title"] // empty' "$tmp")
outName=%[2]s
if [ -n "$title" ]; then outName="$title"; fi
echo "Downloading model weight layer: $layerDigest" >&2
# Fetch blob
if ! oras blob fetch "$ref@$layerDigest" --output "$outName" %[4]s 2>/tmp/oras-blob-error.log; then
	echo "Failed to fetch blob $layerDigest" >&2
	cat /tmp/oras-blob-error.log >&2
	exit 1
fi
ls -l "$outName"
`, artifactURL, modelName, warningMsg, insecureFlag)
	return modelName, cmd
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
func handleHuggingFace(source string, s llb.State) (llb.State, error) {
	// Translate the Hugging Face URL, extracting the branch if provided
	hfURL, modelName, err := ParseHuggingFaceURL(source)
	if err != nil {
		return llb.State{}, err
	}

	// Perform the HTTP download
	opts := []llb.HTTPOption{llb.Filename(modelName)}
	m := llb.HTTP(hfURL, opts...)

	// Determine the model path in the /models directory
	modelPath := fmt.Sprintf("/models/%s", modelName)

	// Copy the downloaded file to the desired location
	s = s.File(
		llb.Copy(m, modelName, modelPath, createCopyOptions()...),
		llb.WithCustomName("Copying "+modelName+" from Hugging Face to "+modelPath),
	)
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
