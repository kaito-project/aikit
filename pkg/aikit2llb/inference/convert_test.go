package inference

import (
	"context"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
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

			legacyInstall := "apt-get install -y rocm && apt-get clean"
			if strings.Contains(combined, legacyInstall) {
				t.Fatalf("expected ROCm install to avoid %q, got: %s", legacyInstall, combined)
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
