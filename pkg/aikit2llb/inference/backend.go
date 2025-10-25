package inference

import (
	"fmt"
	"time"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	defaultBackendName    = "llama-cpp"
	cpuLlamaCppBackend    = "cpu-llama-cpp"
	cuda12LlamaCppBackend = "cuda12-llama-cpp"
)

// getBackendTag returns the appropriate OCI tag for the given backend and runtime.
func getBackendTag(backend, runtime string, platform specs.Platform) string {
	baseTag := localAIVersion

	// Map backend names to their OCI tag equivalents
	backendMap := map[string]string{
		utils.BackendExllamaV2: "exllama2",
		utils.BackendDiffusers: "diffusers",
		utils.BackendLlamaCpp:  "llama-cpp",
	}

	backendName, exists := backendMap[backend]
	if !exists {
		// Default to llama-cpp if backend is not recognized
		backendName = defaultBackendName
	}

	// Handle Apple Silicon - always use CPU llama-cpp
	if runtime == utils.RuntimeAppleSilicon {
		return fmt.Sprintf("%s-cpu-llama-cpp", baseTag)
	}

	// Handle CUDA runtime
	if runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		switch backendName {
		case "exllama2":
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-exllama2", baseTag)
		case "diffusers":
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-diffusers", baseTag)
		case defaultBackendName:
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", baseTag)
		default:
			// Fallback to llama-cpp for unsupported backends
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", baseTag)
		}
	}

	// Handle CPU runtime (default)
	switch backendName {
	case "exllama2":
		return fmt.Sprintf("%s-cpu-exllama2", baseTag)
	case "llama-cpp":
		return fmt.Sprintf("%s-cpu-llama-cpp", baseTag)
	default:
		// For unsupported backends, fallback to llama-cpp
		return fmt.Sprintf("%s-cpu-llama-cpp", baseTag)
	}
}

// getBackendAlias returns the alias name for the backend (used in metadata.json).
func getBackendAlias(backend string) string {
	// Map backend names to their aliases
	aliasMap := map[string]string{
		utils.BackendDiffusers: "diffusers",
		utils.BackendExllamaV2: "exllama2",
		utils.BackendLlamaCpp:  "llama-cpp",
	}

	if alias, exists := aliasMap[backend]; exists {
		return alias
	}
	// Default to llama-cpp for unknown backends
	return "llama-cpp"
}

// getBackendName returns the full backend directory name (used in metadata.json).
func getBackendName(backend, runtime string, platform specs.Platform) string {
	// Handle Apple Silicon - always use cpu-llama-cpp
	if runtime == utils.RuntimeAppleSilicon {
		return cpuLlamaCppBackend
	}

	// Handle CUDA runtime
	if runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		switch backend {
		case utils.BackendExllamaV2:
			return "cuda12-exllama2"
		case utils.BackendDiffusers:
			return "cuda12-diffusers"
		case utils.BackendLlamaCpp:
			return cuda12LlamaCppBackend
		default:
			// Fallback to llama-cpp for unsupported backends
			return cuda12LlamaCppBackend
		}
	}

	// Handle CPU runtime (default)
	switch backend {
	case utils.BackendExllamaV2:
		return "cpu-exllama2"
	case utils.BackendLlamaCpp:
		return cpuLlamaCppBackend
	default:
		// For unsupported backends, fallback to llama-cpp
		return cpuLlamaCppBackend
	}
}

// installBackend downloads and installs a backend from OCI registry.
func installBackend(backend string, c *config.InferenceConfig, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	tag := getBackendTag(backend, c.Runtime, platform)

	// Install dependencies for Python-based backends
	switch backend {
	case utils.BackendExllamaV2:
		merge = installExllamaDependencies(s, merge)
	case utils.BackendDiffusers:
		merge = installPythonBaseDependencies(s, merge)
	case utils.BackendVLLM:
		merge = installPythonBaseDependencies(s, merge)
	}

	// Use Apple Silicon specific registry for arm64 platforms
	var ociImage string
	if runtime := c.Runtime; runtime == utils.RuntimeAppleSilicon && platform.Architecture == utils.PlatformARM64 {
		localAIVersion := "v3.4.0" // temp pin for now
		ociImage = fmt.Sprintf("sertacacr.azurecr.io/llama-cpp:%s-vulkan", localAIVersion)
	} else {
		ociImage = fmt.Sprintf("%s:%s", utils.BackendOCIRegistry, tag)
	}

	// Create the backends directory
	savedState := s
	backendName := getBackendName(backend, c.Runtime, platform)
	backendDir := fmt.Sprintf("/backends/%s", backendName)

	// Download the backend from OCI registry and extract to specific backend directory
	backendState := llb.Image(ociImage, llb.Platform(platform))

	// Copy the backend files to the specific backend directory
	s = s.File(
		llb.Copy(backendState, "/", backendDir+"/", &llb.CopyInfo{
			CreateDestPath: true,
			AllowWildcard:  true,
		}),
		llb.WithCustomName(fmt.Sprintf("Installing backend %s from %s", backend, ociImage)),
	)

	// Ensure the directory exists and create metadata.json for the backend
	backendAlias := getBackendAlias(backend)
	metadataContent := fmt.Sprintf(`{
  "alias": "%s",
  "name": "%s",
  "gallery_url": "github:mudler/LocalAI/backend/index.yaml@master",
  "installed_at": "%s"
}`, backendAlias, backendName, time.Now().UTC().Format(time.RFC3339))

	s = s.File(
		llb.Mkfile(fmt.Sprintf("%s/metadata.json", backendDir), 0o644, []byte(metadataContent)),
		llb.WithCustomName(fmt.Sprintf("Creating metadata.json for backend %s", backendName)),
	)

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

// getDefaultBackends returns the default backends based on runtime if no backends are specified.
func getDefaultBackends(_ string) []string {
	return []string{utils.BackendLlamaCpp}
}

// installBackends installs all specified backends or default backends if none specified.
func installBackends(c *config.InferenceConfig, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	backends := c.Backends
	if len(backends) == 0 {
		backends = getDefaultBackends(c.Runtime)
	}

	for _, backend := range backends {
		merge = installBackend(backend, c, platform, s, merge)

		// For llama-cpp backend with CUDA runtime, also install the CPU version for fallback
		if backend == utils.BackendLlamaCpp && c.Runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
			// Create a modified config with CPU runtime to install the CPU version
			cpuConfig := *c
			cpuConfig.Runtime = "cpu" // Use CPU runtime to force CPU backend installation
			merge = installBackend(backend, &cpuConfig, platform, s, merge)
		}
	}

	return merge
}
