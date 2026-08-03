package inference

import (
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	imageTestInferenceModel  = "test"
	imageTestInferenceSource = "http://test"
)

func TestNewImageConfigEntrypoint(t *testing.T) {
	wrapperPath := "/usr/local/bin/gpu-detect-wrapper"

	tests := []struct {
		name           string
		config         *config.InferenceConfig
		platform       *specs.Platform
		wantEntrypoint []string
	}{
		{
			name: "cuda amd64 standard mode uses local-ai directly",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Config:  imageTestInferenceModel,
				Models:  []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{localAIEntrypointCommand},
		},
		{
			name: "cuda arm64 standard mode uses local-ai directly",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Config:  imageTestInferenceModel,
				Models:  []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{localAIEntrypointCommand},
		},
		{
			name: "cpu standard mode uses local-ai directly",
			config: &config.InferenceConfig{
				Config: imageTestInferenceModel,
				Models: []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{localAIEntrypointCommand},
		},
		{
			name: "cuda amd64 runner mode uses aikit-runner directly",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{runnerEntrypointPath},
		},
		{
			name: "cuda arm64 runner mode uses aikit-runner directly",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{runnerEntrypointPath},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := NewImageConfig(tt.config, tt.platform)

			if !reflect.DeepEqual(img.Config.Entrypoint, tt.wantEntrypoint) {
				t.Errorf("entrypoint = %v, want %v", img.Config.Entrypoint, tt.wantEntrypoint)
			}

			for _, entry := range img.Config.Entrypoint {
				if entry == wrapperPath {
					t.Fatalf("entrypoint should not include legacy GPU wrapper: %v", img.Config.Entrypoint)
				}
			}
		})
	}
}

func TestNewImageConfigEnvironment(t *testing.T) {
	platform := &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux}
	defaultEnv := []string{
		"PATH=" + system.DefaultPathEnv(utils.PlatformLinux),
		"CONFIG_FILE=/config.yaml",
	}
	nvidiaEnv := []string{
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_VISIBLE_DEVICES=all",
		"BUILD_TYPE=cublas",
	}

	tests := []struct {
		name    string
		config  *config.InferenceConfig
		wantEnv []string
	}{
		{
			name: "standard NVIDIA image uses only runtime interface variables",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Models:  []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			wantEnv: append(append([]string{}, defaultEnv...), nvidiaEnv...),
		},
		{
			name: "runner adds Hugging Face cache on models volume",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLM},
			},
			wantEnv: append(append(append([]string{}, defaultEnv...), nvidiaEnv...), runnerHFHomeEnv),
		},
		{
			name: "standard CPU image does not add runner cache",
			config: &config.InferenceConfig{
				Models: []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			wantEnv: append([]string{}, defaultEnv...),
		},
		{
			name: "CPU runner adds Hugging Face cache on models volume",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
			},
			wantEnv: append(append([]string{}, defaultEnv...), runnerHFHomeEnv),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := NewImageConfig(tt.config, platform)
			if !reflect.DeepEqual(img.Config.Env, tt.wantEnv) {
				t.Errorf("environment = %v, want %v", img.Config.Env, tt.wantEnv)
			}

			for _, env := range img.Config.Env {
				if strings.Contains(env, "/usr/local/cuda") || strings.HasPrefix(env, "CUDA_HOME=") {
					t.Errorf("environment should not assume a system CUDA installation: %q", env)
				}
			}
		})
	}
}
