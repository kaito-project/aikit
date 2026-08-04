package inference

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	digest "github.com/opencontainers/go-digest"
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
			name:    "CPU diffusers falls back to v4 CPU llama-cpp",
			backend: utils.BackendDiffusers,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-cpu-llama-cpp", localAILlamaCppBackendVersion),
		},
		{
			name:    "CPU vllm falls back to v4 CPU llama-cpp",
			backend: utils.BackendVLLM,
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
			want: fmt.Sprintf("%s-gpu-nvidia-cuda-12-vllm", localAIBinaryVersion),
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
			name:    "ROCm llama-cpp",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeROCm,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: fmt.Sprintf("%s-gpu-rocm-hipblas-llama-cpp", localAIROCmBackendVersion),
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
		name     string
		backend  string
		runtime  string
		platform specs.Platform
		want     string
	}{
		{
			name:    "llama-cpp defaults to v4 backend tags",
			backend: utils.BackendLlamaCpp,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILlamaCppBackendVersion,
		},
		{
			name:    "CPU diffusers falls back to v4 backend tags",
			backend: utils.BackendDiffusers,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILlamaCppBackendVersion,
		},
		{
			name:    "CPU vllm falls back to v4 backend tags",
			backend: utils.BackendVLLM,
			runtime: "",
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILlamaCppBackendVersion,
		},
		{
			name:    "diffusers stays on legacy backend tags",
			backend: utils.BackendDiffusers,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILegacyBackendVersion,
		},
		{
			name:    "vllm uses current backend tags",
			backend: utils.BackendVLLM,
			runtime: utils.RuntimeNVIDIA,
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name:    "apple silicon stays on legacy backend tags",
			backend: utils.BackendLlamaCpp,
			runtime: utils.RuntimeAppleSilicon,
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: localAILegacyBackendVersion,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBackendVersion(tt.backend, tt.runtime, tt.platform)
			if got != tt.want {
				t.Errorf("getBackendVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetLocalAIArtifactVersion(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.InferenceConfig
		platform specs.Platform
		want     string
	}{
		{
			name: "default llama-cpp uses current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime: "",
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name: "vllm uses current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLM},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name: "diffusers uses legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendDiffusers},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILegacyBackendVersion,
		},
		{
			name: "CPU diffusers falls back to current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  "",
				Backends: []string{utils.BackendDiffusers},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name: "CPU vllm falls back to current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  "",
				Backends: []string{utils.BackendVLLM},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name: "apple silicon stays on legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeAppleSilicon,
			},
			platform: specs.Platform{
				Architecture: utils.PlatformARM64,
			},
			want: localAILegacyBackendVersion,
		},
		{
			name: "llama-cpp and vllm use current LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp, utils.BackendVLLM},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAIBinaryVersion,
		},
		{
			name: "mixed current and legacy backends choose legacy LocalAI binary",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLM, utils.BackendDiffusers},
			},
			platform: specs.Platform{
				Architecture: utils.PlatformAMD64,
			},
			want: localAILegacyBackendVersion,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getLocalAIArtifactVersion(tt.config, tt.platform)
			if got != tt.want {
				t.Errorf("getLocalAIArtifactVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInstallBackendVLLMHasOptimizedCopyWithoutCompatibilityPatch(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{
		Runtime:  utils.RuntimeNVIDIA,
		Backends: []string{utils.BackendVLLM},
	}
	base := llb.Image(utils.UbuntuBase, llb.Platform(platform))
	state := installBackend(utils.BackendVLLM, cfg, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal vLLM backend definition: %v", err)
	}

	backendDir := "/backends/cuda12-vllm"
	var backendFileOpCount int
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}

		if metadata, ok := definition.Metadata[digest.FromBytes(data)]; ok {
			customName := metadata.Description["llb.customname"]
			if strings.Contains(customName, "Patching vLLM backend") {
				t.Fatalf("vLLM definition contains compatibility patch op %q", customName)
			}
		}

		if exec := op.GetExec(); exec != nil {
			command := strings.Join(exec.Meta.Args, "\x00")
			if strings.Contains(command, "flash_attn") || strings.Contains(command, "get_model_config()") {
				t.Fatalf("vLLM definition contains obsolete compatibility patch command %q", command)
			}
		}

		fileOp := op.GetFile()
		if fileOp == nil {
			continue
		}

		var backendCopy, backendMetadata bool
		for _, action := range fileOp.Actions {
			if copyAction := action.GetCopy(); copyAction != nil && copyAction.Src == "/" && strings.HasPrefix(copyAction.Dest, backendDir) {
				backendCopy = true
				if copyAction.AllowWildcard {
					t.Fatal("backend root copy unexpectedly enables wildcard handling")
				}
			}
			if mkfile := action.GetMkfile(); mkfile != nil && mkfile.Path == backendDir+"/metadata.json" {
				backendMetadata = true
			}
		}

		if backendCopy || backendMetadata {
			backendFileOpCount++
			if !backendCopy || !backendMetadata {
				t.Fatal("backend root copy and metadata creation are not chained in the same file op")
			}
		}
	}

	if backendFileOpCount != 1 {
		t.Fatalf("vLLM backend file op count = %d, want 1", backendFileOpCount)
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
