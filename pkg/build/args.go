package build

import (
	"fmt"
	"path"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/utils"
)

// parseBuildArgs parses the build arguments and configures inference settings.
func parseBuildArgs(opts map[string]string, inferenceCfg *config.InferenceConfig) error {
	if inferenceCfg == nil {
		return nil
	}

	// Get model and runtime arguments
	modelArg := getBuildArg(opts, "model")
	runtimeArg := getBuildArg(opts, "runtime")

	// Set the runtime if provided
	if runtimeArg != "" {
		inferenceCfg.Runtime = runtimeArg
	}

	// Set the model if provided
	if modelArg != "" {
		var modelName, modelSource string
		var err error

		// Handle based on the URL prefix
		switch {
		case strings.HasPrefix(modelArg, "huggingface://"):
			// Handle Hugging Face URLs with optional branch
			modelSource, modelName, err = inference.ParseHuggingFaceURL(modelArg)
			if err != nil {
				return err
			}

		case strings.HasPrefix(modelArg, "http://"), strings.HasPrefix(modelArg, "https://"):
			// Handle HTTP(S) URLs directly
			modelName = utils.FileNameFromURL(modelArg)
			modelSource = modelArg

		case strings.HasPrefix(modelArg, "oci://"):
			// Handle OCI URLs
			modelName = parseOCIURL(modelArg)
			modelSource = modelArg

		default:
			// Assume it's a local file path
			modelName = path.Base(modelArg)
			modelSource = modelArg
		}

		// Set the inference configuration
		inferenceCfg.Models = []config.Model{
			{
				Name:   modelName,
				Source: modelSource,
			},
		}
		inferenceCfg.Config = generateInferenceConfig(modelName)
	}

	return nil
}

// generateInferenceConfig generates the inference configuration for the given model name.
func generateInferenceConfig(modelName string) string {
	return fmt.Sprintf(`
- name: %[1]s
  backend: llama
  parameters:
    model: %[1]s`, modelName)
}

// parseOCIURL extracts model name for OCI-based models.
func parseOCIURL(source string) string {
	const ollamaRegistryURL = "registry.ollama.ai"
	artifactURL := strings.TrimPrefix(source, "oci://")
	var modelName string

	if strings.HasPrefix(artifactURL, ollamaRegistryURL) {
		// Special handling for Ollama registry
		artifactURLWithoutTag := strings.Split(artifactURL, ":")[0]
		modelName = strings.Split(artifactURLWithoutTag, "/")[2]
	} else {
		// Generic OCI artifact
		modelName = path.Base(artifactURL)
		modelName = strings.Split(modelName, ":")[0]
		modelName = strings.Split(modelName, "@")[0]
	}

	return modelName
}
