package inference

import (
	stderrors "errors"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
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
			img, err := NewImageConfig(tt.config, tt.platform)
			if err != nil {
				t.Fatalf("NewImageConfig() error = %v", err)
			}

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
	standardOfflineEnv := []string{
		localAIModelGalleriesEnv,
		localAIBackendGalleriesEnv,
		localAIDisableModelGalleryAutoloadEnv,
		localAIDisableBackendGalleryAutoloadEnv,
		localAIDisableGalleryWarmupEnv,
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
			wantEnv: append(
				append(append([]string{}, defaultEnv...), catalogEnv...),
				standardOfflineEnv...,
			),
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
			wantEnv: append(append([]string{}, defaultEnv...), standardOfflineEnv...),
		},
		{
			name: "load-to-memory precedes catalog environment",
			config: &config.InferenceConfig{
				Models:       []config.Model{{Name: imageTestInferenceModel, Source: imageTestInferenceSource}},
				LoadToMemory: []string{"chat", "embeddings"},
			},
			environment: catalogEnv,
			wantEnv: append(
				append(
					append(append([]string{}, defaultEnv...), localAILoadToMemoryEnv+"chat,embeddings"),
					catalogEnv...,
				),
				standardOfflineEnv...,
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

func TestNewImageConfigUsesOnlyPublicRuntimeLabel(t *testing.T) {
	platform := &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux}
	backend := testArbitraryBackendPlan(*platform)
	tests := []struct {
		name        string
		configValue string
		wantRuntime backendcatalog.Runtime
	}{
		{name: "omitted runtime is CPU", wantRuntime: backendcatalog.RuntimeCPU},
		{name: "explicit CPU", configValue: string(backendcatalog.RuntimeCPU), wantRuntime: backendcatalog.RuntimeCPU},
		{name: "CUDA alias remains CUDA", configValue: string(backendcatalog.RuntimeCUDA), wantRuntime: backendcatalog.RuntimeCUDA},
		{name: "exact CUDA 12", configValue: string(backendcatalog.RuntimeCUDA12), wantRuntime: backendcatalog.RuntimeCUDA12},
		{name: "exact CUDA 13", configValue: string(backendcatalog.RuntimeCUDA13), wantRuntime: backendcatalog.RuntimeCUDA13},
		{name: "ROCm", configValue: string(backendcatalog.RuntimeROCm), wantRuntime: backendcatalog.RuntimeROCm},
		{name: "Apple Silicon", configValue: string(backendcatalog.RuntimeAppleSilicon), wantRuntime: backendcatalog.RuntimeAppleSilicon},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			img := NewImageConfigWithBackend(&config.InferenceConfig{Runtime: test.configValue}, backend, platform)
			if got := img.Config.Labels["ai.kaito.aikit.runtime"]; got != string(test.wantRuntime) {
				t.Errorf("runtime label = %q, want %q", got, test.wantRuntime)
			}
			if _, ok := img.Config.Labels["ai.kaito.aikit.backend.selector"]; ok {
				t.Error("image labels unexpectedly expose the internal catalog selector")
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
			img, err := NewImageConfig(tt.config, platform)
			if err != nil {
				t.Fatalf("NewImageConfig() error = %v", err)
			}
			if !reflect.DeepEqual(img.Config.Cmd, tt.wantCmd) {
				t.Errorf("command = %v, want %v", img.Config.Cmd, tt.wantCmd)
			}
		})
	}
}

func TestNewImageConfigReturnsResolutionError(t *testing.T) {
	platform := &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux}
	image, err := NewImageConfig(&config.InferenceConfig{
		Runtime:  utils.RuntimeCUDA12,
		Backends: []string{utils.BackendVLLMCpp},
	}, platform)
	if err == nil {
		t.Fatal("NewImageConfig() succeeded, want error")
	}
	if image != nil {
		t.Fatalf("NewImageConfig() image = %#v, want nil", image)
	}
	if !stderrors.Is(err, backendcatalog.ErrNotFound) {
		t.Fatalf("NewImageConfig() error = %v, want exact resolution failure", err)
	}
	if !strings.Contains(err.Error(), "resolving backend for image config") {
		t.Fatalf("NewImageConfig() error = %q, want image config context", err)
	}
}

func TestNewImageConfigRequiresPlatform(t *testing.T) {
	image, err := NewImageConfig(&config.InferenceConfig{}, nil)
	if err == nil {
		t.Fatal("NewImageConfig() succeeded, want error")
	}
	if image != nil {
		t.Fatalf("NewImageConfig() image = %#v, want nil", image)
	}
	if err.Error() != "platform is required" {
		t.Fatalf("NewImageConfig() error = %q, want platform requirement", err)
	}
}
