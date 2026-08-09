package inference

import (
	"reflect"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	imageTestDebugArgument     = "--debug"
	imageTestInferenceModel    = "test"
	imageTestInferenceSource   = "http://test"
	imageTestLoadToMemoryModel = "model"
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

func TestNewImageConfigEnvironmentUsesCatalogPlan(t *testing.T) {
	platform := &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux}
	defaultEnv := []string{
		"PATH=" + system.DefaultPathEnv(utils.PlatformLinux),
		"CONFIG_FILE=/config.yaml",
	}
	catalogEnv := []string{"ARBITRARY_ACCELERATOR=enabled", "ARBITRARY_CACHE=/var/cache/arbitrary"}

	tests := []struct {
		name        string
		config      *config.InferenceConfig
		environment []string
		wantEnv     []string
	}{
		{
			name: "standard image uses catalog environment",
			config: &config.InferenceConfig{
				Models: []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			environment: catalogEnv,
			wantEnv:     append(append([]string{}, defaultEnv...), catalogEnv...),
		},
		{
			name: "runner appends Hugging Face cache after catalog environment",
			config: &config.InferenceConfig{
				Backends: []string{testArbitraryFamily},
			},
			environment: catalogEnv,
			wantEnv:     append(append(append([]string{}, defaultEnv...), catalogEnv...), runnerHFHomeEnv),
		},
		{
			name: "standard image with empty plan environment",
			config: &config.InferenceConfig{
				Models: []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
			},
			wantEnv: defaultEnv,
		},
		{
			name: "load-to-memory precedes catalog environment",
			config: &config.InferenceConfig{
				Models:       []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
				LoadToMemory: []string{"chat", "embeddings"},
			},
			environment: catalogEnv,
			wantEnv: append(
				append(append([]string{}, defaultEnv...), localAILoadToMemoryEnv+"chat,embeddings"),
				catalogEnv...,
			),
		},
		{
			name: "runner retains load-to-memory and catalog environment",
			config: &config.InferenceConfig{
				Backends:     []string{testArbitraryFamily},
				LoadToMemory: []string{imageTestLoadToMemoryModel},
			},
			environment: catalogEnv,
			wantEnv: append(
				append(
					append(append([]string{}, defaultEnv...), localAILoadToMemoryEnv+imageTestLoadToMemoryModel),
					catalogEnv...,
				),
				runnerHFHomeEnv,
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := testArbitraryBackendPlan(*platform)
			backend.Environment = tt.environment
			img := NewImageConfigWithBackend(tt.config, backend, platform)
			if !reflect.DeepEqual(img.Config.Env, tt.wantEnv) {
				t.Errorf("environment = %v, want %v", img.Config.Env, tt.wantEnv)
			}
		})
	}
}

func TestNewImageConfigCommandWithLoadToMemory(t *testing.T) {
	platform := &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux}

	tests := []struct {
		name    string
		config  *config.InferenceConfig
		wantCmd []string
	}{
		{
			name: "standard image command is unchanged",
			config: &config.InferenceConfig{
				Debug:        true,
				Config:       "config",
				Models:       []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
				LoadToMemory: []string{imageTestLoadToMemoryModel},
			},
			wantCmd: []string{imageTestDebugArgument, "--config-file=/config.yaml"},
		},
		{
			name: "runner command remains empty",
			config: &config.InferenceConfig{
				Backends:     []string{utils.BackendLlamaCpp},
				LoadToMemory: []string{imageTestLoadToMemoryModel},
			},
			wantCmd: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := NewImageConfig(tt.config, platform)
			if !reflect.DeepEqual(img.Config.Cmd, tt.wantCmd) {
				t.Errorf("command = %v, want %v", img.Config.Cmd, tt.wantCmd)
			}
		})
	}
}
