package inference

import (
	"fmt"
	"testing"

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
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "CPU exllama2",
			backend: utils.BackendExllamaV2,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-exllama2", localAIVersion),
		},
		{
			name:    "CUDA llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAIVersion),
		},
		{
			name:    "CUDA exllama2",
			backend: utils.BackendExllamaV2,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-exllama2", localAIVersion),
		},
		{
			name:    "CUDA diffusers",
			backend: utils.BackendDiffusers,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-diffusers", localAIVersion),
		},
		{
			name:    "Apple Silicon always uses CPU llama-cpp",
			backend: utils.BackendExllamaV2,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "Apple Silicon llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "Unsupported backend falls back to CPU llama-cpp",
			backend: "unknown",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "CUDA unsupported backend falls back to CUDA llama-cpp",
			backend: "unknown",
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAIVersion),
		},
		{
			name:    "Unknown backend falls back to CPU llama-cpp",
			backend: "unknown",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "Empty backend name defaults to CPU llama-cpp",
			backend: "",
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAIVersion),
		},
		{
			name:    "Empty backend with CUDA runtime defaults to CUDA llama-cpp",
			backend: "",
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-llama-cpp", localAIVersion),
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
			name:    "exllama2 backend",
			backend: utils.BackendExllamaV2,
			want:    "exllama2",
		},
		{
			name:    "llama-cpp backend",
			backend: utils.BackendLlamaCpp,
			want:    "llama-cpp",
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
			name:    "CPU exllama2",
			backend: utils.BackendExllamaV2,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cpu-exllama2",
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
			name:    "CUDA exllama2",
			backend: utils.BackendExllamaV2,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: "cuda12-exllama2",
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
			name:    "Apple Silicon always uses cpu-llama-cpp regardless of backend",
			backend: utils.BackendExllamaV2,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: "cpu-llama-cpp",
		},
		{
			name:    "Apple Silicon llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: "cpu-llama-cpp",
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
		{
			name:    "ARM64 with CPU runtime - exllama2 returns cpu-exllama2",
			backend: utils.BackendExllamaV2,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: "cpu-exllama2",
		},
		{
			name:    "ARM64 with NVIDIA runtime (edge case) - exllama2 returns cpu-exllama2",
			backend: utils.BackendExllamaV2,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: "cpu-exllama2",
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
