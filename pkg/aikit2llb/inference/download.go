package inference

import (
	"errors"
	"fmt"
	"os"
	"path"
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
	var orasCmd, modelName string

	if strings.HasPrefix(artifactURL, ollamaRegistryURL) {
		// Handle specific registry case
		modelName, orasCmd = handleOllamaRegistry(artifactURL)
	} else {
		// Handle generic OCI artifact
		modelName = extractModelName(artifactURL)
		orasCmd = fmt.Sprintf("oras blob fetch %[1]s --output /models/%[2]s", artifactURL, modelName)
	}

	// Install jq and execute the oras command
	toolingImage = toolingImage.Run(utils.Shf("apk add jq curl && %s", orasCmd)).Root()

	modelPath := fmt.Sprintf("/models/%s", modelName)

	s = s.File(
		llb.Copy(toolingImage, modelName, modelPath, createCopyOptions()...),
		llb.WithCustomName("Copying "+artifactURL+" to "+modelPath),
	)
	return s
}

// handleOllamaRegistry handles the Ollama registry specific download.
func handleOllamaRegistry(artifactURL string) (string, string) {
	artifactURLWithoutTag := strings.Split(artifactURL, ":")[0]
	tag := strings.Split(artifactURL, ":")[1]
	modelName := strings.Split(artifactURLWithoutTag, "/")[2]
	orasCmd := fmt.Sprintf("oras blob fetch %[1]s@$(curl https://%[2]s/v2/library/%[3]s/manifests/%[4]s | jq -r '.layers[] | select(.mediaType == \"application/vnd.ollama.image.model\").digest') --output %[2]s", artifactURLWithoutTag, ollamaRegistryURL, modelName, tag)
	return modelName, orasCmd
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

// parseHuggingFaceURL converts a huggingface:// URL to https:// URL with optional branch support.
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
