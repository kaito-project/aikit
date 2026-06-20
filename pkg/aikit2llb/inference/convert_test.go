package inference

import (
	"context"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestInstallRocmInstallsPciutilsForLlamaCpp(t *testing.T) {
	tests := []struct {
		name     string
		backends []string
	}{
		{
			name:     "implicit default llama-cpp backend",
			backends: nil,
		},
		{
			name:     "explicit llama-cpp backend",
			backends: []string{utils.BackendLlamaCpp},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.InferenceConfig{
				Runtime:  utils.RuntimeROCm,
				Backends: tt.backends,
			}

			base := llb.Image(utils.Ubuntu24Base)
			_, merged := installRocm(cfg, base, base)

			def, err := merged.Marshal(context.Background())
			if err != nil {
				t.Fatalf("marshal failed: %v", err)
			}

			combined := marshalDefinitionToString(def)
			wantInstall := "apt-get install -y pciutils rocm && apt-get clean"
			if !strings.Contains(combined, wantInstall) {
				t.Fatalf("expected ROCm install to contain %q, got: %s", wantInstall, combined)
			}
		})
	}
}

func marshalDefinitionToString(def *llb.Definition) string {
	if def == nil {
		return ""
	}

	var combined strings.Builder
	for _, d := range def.ToPB().Def {
		combined.Write(d)
	}

	return combined.String()
}

// TestCopyModelsAbsoluteLocalPath guards the scheme-dispatch fix: an absolute
// local model path (no URI scheme) must be treated as a local file, not
// rejected. The previous url.ParseRequestURI guard caused absolute paths to
// fall through to a hard "unsupported URL scheme" error.
func TestCopyModelsAbsoluteLocalPath(t *testing.T) {
	cfg := &config.InferenceConfig{
		Runtime: "",
		Models: []config.Model{
			{Name: "local", Source: "/models/local.gguf"},
		},
	}

	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	base := llb.Image(utils.UbuntuBase)
	state, merged, err := copyModels(cfg, base, base, platform)
	if err != nil {
		t.Fatalf("copyModels returned error for absolute local path: %v", err)
	}

	if _, err := merged.Marshal(context.Background()); err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	_ = state
}
