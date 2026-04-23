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
	vulkanLlamaCppBackend = "gpu-vulkan-llama-cpp"
)

func normalizeBackend(backend string) string {
	switch backend {
	case utils.BackendDiffusers, utils.BackendLlamaCpp, utils.BackendVLLM:
		return backend
	default:
		return defaultBackendName
	}
}

// getEffectiveBackend resolves the backend that will actually be installed for
// the requested runtime/platform combination after any fallback.
func getEffectiveBackend(backend, runtime string, platform specs.Platform) string {
	normalizedBackend := normalizeBackend(backend)

	if runtime == utils.RuntimeAppleSilicon {
		return defaultBackendName
	}

	if runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		return normalizedBackend
	}

	return defaultBackendName
}

// getBackendVersion returns the backend OCI tag version to use for the
// effective runtime/platform backend. Keep non-llama and Apple Silicon
// backends pinned to the legacy tag until matching v4 artifacts are mirrored
// upstream.
func getBackendVersion(backend, runtime string, platform specs.Platform) string {
	if runtime == utils.RuntimeAppleSilicon {
		return localAILegacyBackendVersion
	}

	if getEffectiveBackend(backend, runtime, platform) == defaultBackendName {
		return localAILlamaCppBackendVersion
	}

	return localAILegacyBackendVersion
}

// getLocalAIArtifactVersion returns the LocalAI artifact version to install for
// the image. If any configured backend still requires legacy compatibility, use
// the legacy LocalAI binary so legacy-pinned backends are not paired with v4.
func getLocalAIArtifactVersion(c *config.InferenceConfig, platform specs.Platform) string {
	backends := c.Backends
	if len(backends) == 0 {
		backends = getDefaultBackends(c.Runtime)
	}

	for _, backend := range backends {
		if getBackendVersion(backend, c.Runtime, platform) == localAILegacyBackendVersion {
			return localAILegacyBackendVersion
		}
	}

	return localAIBinaryVersion
}

// getBackendTag returns the appropriate OCI tag for the given backend and runtime.
func getBackendTag(backend, runtime string, platform specs.Platform) string {
	baseTag := getBackendVersion(backend, runtime, platform)
	backendName := getEffectiveBackend(backend, runtime, platform)

	// Handle Apple Silicon - use Vulkan llama-cpp.
	if runtime == utils.RuntimeAppleSilicon {
		return fmt.Sprintf("%s-%s", baseTag, vulkanLlamaCppBackend)
	}

	// Handle CUDA runtime
	if runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		switch backendName {
		case "diffusers":
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-diffusers", baseTag)
		case "vllm":
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-vllm", baseTag)
		case defaultBackendName:
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", baseTag)
		default:
			// Fallback to llama-cpp for unsupported backends
			return fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", baseTag)
		}
	}

	// Handle ROCm runtime.
	if runtime == utils.RuntimeROCm && platform.Architecture == utils.PlatformAMD64 {
		return fmt.Sprintf("%s-gpu-rocm-hipblas-llama-cpp", localAIROCmBackendVersion)
	}

	// Handle CPU runtime (default).
	return fmt.Sprintf("%s-cpu-llama-cpp", baseTag)
}

// getBackendAlias returns the alias name for the backend (used in metadata.json).
func getBackendAlias(backend string) string {
	return normalizeBackend(backend)
}

// getBackendName returns the full backend directory name (used in metadata.json).
func getBackendName(backend, runtime string, platform specs.Platform) string {
	// Handle Apple Silicon - use Vulkan llama-cpp
	if runtime == utils.RuntimeAppleSilicon {
		return vulkanLlamaCppBackend
	}

	// Handle CUDA runtime
	if runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		switch getEffectiveBackend(backend, runtime, platform) {
		case utils.BackendDiffusers:
			return "cuda12-diffusers"
		case utils.BackendVLLM:
			return "cuda12-vllm"
		case defaultBackendName:
			return cuda12LlamaCppBackend
		default:
			// Fallback to llama-cpp for unsupported backends
			return cuda12LlamaCppBackend
		}
	}

	// Handle ROCm runtime
	if runtime == utils.RuntimeROCm && platform.Architecture == utils.PlatformAMD64 {
		// Only llama-cpp backend is supported for ROCm
		return "hipblas-llama-cpp"
	}

	// Handle CPU runtime (default)
	return cpuLlamaCppBackend
}

// installBackend downloads and installs a backend from OCI registry.
func installBackend(backend string, c *config.InferenceConfig, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	tag := getBackendTag(backend, c.Runtime, platform)

	// Install dependencies for Python-based backends
	if backend == utils.BackendDiffusers {
		merge = installDiffusersDependencies(s, merge)
	}
	if backend == utils.BackendVLLM {
		merge = installVLLMDependencies(s, merge)
	}

	// Build the OCI image reference
	ociImage := fmt.Sprintf("%s:%s", utils.BackendOCIRegistry, tag)

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

	// Apply workarounds for the pre-built vLLM backend image.
	if backend == utils.BackendVLLM {
		// Remove broken flash_attn package (PyTorch ABI incompatibility).
		// Patch backend.py to use the current vLLM AsyncLLM API
		// (get_model_config() was replaced by the model_config property).
		s = s.Run(utils.Shf(
			"rm -rf %[1]s/venv/lib/python*/site-packages/flash_attn* && "+
				"sed -i 's/await self.llm.get_model_config()/self.llm.model_config/' %[1]s/backend.py",
			backendDir),
			llb.WithCustomNamef("Patching vLLM backend %s for compatibility", backendName),
		).Root()
	}

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

		// For llama-cpp backend with ROCm runtime, also install the CPU version for fallback
		if backend == utils.BackendLlamaCpp && c.Runtime == utils.RuntimeROCm && platform.Architecture == utils.PlatformAMD64 {
			// Create a modified config with CPU runtime to install the CPU version
			cpuConfig := *c
			cpuConfig.Runtime = "cpu" // Use CPU runtime to force CPU backend installation
			merge = installBackend(backend, &cpuConfig, platform, s, merge)
		}
	}

	return merge
}
