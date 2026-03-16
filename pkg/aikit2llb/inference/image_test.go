package inference

import (
	"reflect"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
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
				Config:  "test",
				Models:  []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{"local-ai"},
		},
		{
			name: "cuda arm64 standard mode uses local-ai directly",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
				Config:  "test",
				Models:  []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{"local-ai"},
		},
		{
			name: "cpu standard mode uses local-ai directly",
			config: &config.InferenceConfig{
				Config: "test",
				Models: []config.Model{{Name: "test", Source: "http://test"}},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{"local-ai"},
		},
		{
			name: "cuda amd64 runner mode uses aikit-runner directly",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{"/usr/local/bin/aikit-runner"},
		},
		{
			name: "cuda arm64 runner mode uses aikit-runner directly",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendLlamaCpp},
			},
			platform:       &specs.Platform{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			wantEntrypoint: []string{"/usr/local/bin/aikit-runner"},
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
