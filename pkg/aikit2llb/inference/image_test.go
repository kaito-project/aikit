package inference

import (
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestNewImageConfig_GPUWrapper(t *testing.T) {
	wrapperPath := "/usr/local/bin/gpu-detect-wrapper"

	tests := []struct {
		name          string
		config        *config.InferenceConfig
		platform      *specs.Platform
		wantWrapper   bool
		wantEntrySize int // expected number of entrypoint elements
	}{
		{
			name: "cuda amd64 standard mode gets wrapper",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Config:  "test",
				Models:  []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:      &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantWrapper:   true,
			wantEntrySize: 2,
		},
		{
			name: "cuda arm64 standard mode does not get wrapper",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Config:  "test",
				Models:  []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:      &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantWrapper:   false,
			wantEntrySize: 1,
		},
		{
			name: "cpu runtime amd64 does not get wrapper",
			config: &config.InferenceConfig{
				Config: "test",
				Models: []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:      &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantWrapper:   false,
			wantEntrySize: 1,
		},
		{
			name: "cuda amd64 runner mode gets wrapper",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:      &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantWrapper:   true,
			wantEntrySize: 2,
		},
		{
			name: "cuda arm64 runner mode does not get wrapper",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:      &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantWrapper:   false,
			wantEntrySize: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := NewImageConfig(tt.config, tt.platform)
			entrypoint := img.Config.Entrypoint

			if len(entrypoint) != tt.wantEntrySize {
				t.Errorf("entrypoint length = %d, want %d; entrypoint = %v", len(entrypoint), tt.wantEntrySize, entrypoint)
			}

			hasWrapper := len(entrypoint) > 0 && entrypoint[0] == wrapperPath
			if hasWrapper != tt.wantWrapper {
				t.Errorf("wrapper present = %v, want %v; entrypoint = %v", hasWrapper, tt.wantWrapper, entrypoint)
			}
		})
	}
}

func TestGPUDetectionWrapperRespectsEnvOverride(t *testing.T) {
	// Verify the wrapper script contains the env var check early-exit
	if !containsSubstring(gpuDetectionWrapper, `if [ -n "$LOCALAI_FORCE_META_BACKEND_CAPABILITY" ]; then`) {
		t.Error("wrapper script should check for existing LOCALAI_FORCE_META_BACKEND_CAPABILITY and exit early")
	}
}

func TestGPUDetectionWrapperChecksMultipleSources(t *testing.T) {
	// Verify the wrapper checks /dev/nvidiactl, nvidia-smi, and lspci
	checks := []string{"/dev/nvidiactl", "nvidia-smi", "lspci"}
	for _, check := range checks {
		if !containsSubstring(gpuDetectionWrapper, check) {
			t.Errorf("wrapper script should check %s for GPU detection", check)
		}
	}
}

func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
