package inference

import (
	"fmt"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestGetBackendTag(t *testing.T) {
	tests := []struct {
		name     string
		backend  string
		runtime  string
		platform specs.Platform
		want     string
	}{
		{
			name:    "CPU llama-cpp default",
			backend: utils.BackendLlamaCpp,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "CUDA llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "CUDA diffusers",
			backend: utils.BackendDiffusers,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-diffusers", localAILegacyBackendVersion),
		},
		{
			name:    "CUDA vllm",
			backend: utils.BackendVLLM,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-vllm", localAILegacyBackendVersion),
		},
		{
			name:    "Apple Silicon llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: fmt.Sprintf("%s-gpu-vulkan-llama-cpp", localAILegacyBackendVersion),
		},
		{
			name:    "Unsupported backend falls back to CPU llama-cpp",
			backend: "unknown",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "CUDA unsupported backend falls back to CUDA llama-cpp",
			backend: "unknown",
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "Empty backend name defaults to CPU llama-cpp",
			backend: "",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "Empty backend with CUDA runtime defaults to CUDA llama-cpp",
			backend: "",
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAILlamaCppBackendVersion),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBackendTag(tt.backend, tt.runtime, tt.platform)
			if got != tt.want {
				t.Errorf("getBackendTag() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetBackendVersion(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		runtime string
		want    string
	}{
		{
			name:    "llama-cpp defaults to v4 backend tags",
			backend: utils.BackendLlamaCpp,
			runtime: "",
			want:    localAILlamaCppBackendVersion,
		},
		{
			name:    "diffusers stays on legacy backend tags",
			backend: utils.BackendDiffusers,
			runtime: utils.RuntimeNVIDIA,
			want:    localAILegacyBackendVersion,
		},
		{
			name:    "vllm stays on legacy backend tags",
			backend: utils.BackendVLLM,
			runtime: utils.RuntimeNVIDIA,
			want:    localAILegacyBackendVersion,
		},
		{
			name:    "apple silicon stays on legacy backend tags",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			want:    localAILegacyBackendVersion,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBackendVersion(tt.backend, tt.runtime)
			if got != tt.want {
				t.Errorf("getBackendVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetLocalAIArtifactVersion(t *testing.T) {
	tests := []struct {
		name   string
		config *config.InferenceConfig
		want   string
	}{
		{
			name: "default llama-cpp uses current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime: "",
			},
			want: localAIBinaryVersion,
		},
		{
			name: "vllm uses legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLM},
			},
			want: localAILegacyBackendVersion,
		},
		{
			name: "diffusers uses legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendDiffusers},
			},
			want: localAILegacyBackendVersion,
		},
		{
			name: "apple silicon stays on legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeAppleSilicon,
			},
			want: localAILegacyBackendVersion,
		},
		{
			name: "mixed backends choose legacy LocalAI binary when needed",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp, utils.BackendVLLM},
			},
			want: localAILegacyBackendVersion,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getLocalAIArtifactVersion(tt.config)
			if got != tt.want {
				t.Errorf("getLocalAIArtifactVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetDefaultBackends(t *testing.T) {
	tests := []struct {
		name    string
		runtime string
		want    []string
	}{
		{
			name:    "empty runtime (CPU) defaults to llama-cpp",
			runtime: "",
			want:    []string{utils.BackendLlamaCpp},
		},
		{
			name:    "CUDA runtime defaults to llama-cpp",
			runtime: utils.RuntimeNVIDIA,
			want:    []string{utils.BackendLlamaCpp},
		},
		{
			name:    "Apple Silicon runtime defaults to llama-cpp",
			runtime: utils.RuntimeAppleSilicon,
			want:    []string{utils.BackendLlamaCpp},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getDefaultBackends(tt.runtime)
			if len(got) != len(tt.want) {
				t.Errorf("getDefaultBackends() = %v, want %v", got, tt.want)
				return
			}
			for i, backend := range got {
				if backend != tt.want[i] {
					t.Errorf("getDefaultBackends()[%d] = %v, want %v", i, backend, tt.want[i])
				}
			}
		})
	}
}

func TestGetBackendAlias(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		want    string
	}{
		{
			name:    "diffusers backend",
			backend: utils.BackendDiffusers,
			want:    "diffusers",
		},
		{
			name:    "llama-cpp backend",
			backend: utils.BackendLlamaCpp,
			want:    "llama-cpp",
		},
		{
			name:    "vllm backend",
			backend: utils.BackendVLLM,
			want:    "vllm",
		},
		{
			name:    "unknown backend defaults to llama-cpp",
			backend: "unknown",
			want:    "llama-cpp",
		},
		{
			name:    "empty backend defaults to llama-cpp",
			backend: "",
			want:    "llama-cpp",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBackendAlias(tt.backend)
			if got != tt.want {
				t.Errorf("getBackendAlias() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetBackendName(t *testing.T) {
	tests := []struct {
		name     string
		backend  string
		runtime  string
		platform specs.Platform
		want     string
	}{
		{
			name:    "CPU llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cpu-llama-cpp",
		},
		{
			name:    "CUDA llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cuda12-llama-cpp",
		},
		{
			name:    "CUDA diffusers",
			backend: utils.BackendDiffusers,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cuda12-diffusers",
		},
		{
			name:    "CUDA vllm",
			backend: utils.BackendVLLM,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cuda12-vllm",
		},
		{
			name:    "Apple Silicon llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: "gpu-vulkan-llama-cpp",
		},
		{
			name:    "Unknown backend on CPU defaults to cpu-llama-cpp",
			backend: "unknown",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cpu-llama-cpp",
		},
		{
			name:    "Unknown backend on CUDA defaults to cuda12-llama-cpp",
			backend: "unknown",
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cuda12-llama-cpp",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBackendName(tt.backend, tt.runtime, tt.platform)
			if got != tt.want {
				t.Errorf("getBackendName() = %v, want %v", got, tt.want)
			}
		})
	}
}
