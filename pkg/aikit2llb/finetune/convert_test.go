package finetune

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/solver/pb"
	"gopkg.in/yaml.v2"
)

func TestAikit2LLBWritesConfigWithoutShellInterpolation(t *testing.T) {
	const marker = `$(touch /tmp/aikit-injection)`
	cfg := &config.FineTuneConfig{
		Target:    utils.TargetUnsloth,
		BaseModel: "model-with-shell-text-" + marker + "\nand-a-newline",
		Output: config.FineTuneOutputSpec{
			Name:     "test-model",
			Quantize: "q4_k_m",
		},
	}

	wantConfig, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal expected config: %v", err)
	}

	state, err := Aikit2LLB(cfg)
	if err != nil {
		t.Fatalf("Aikit2LLB returned an error: %v", err)
	}
	def, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal LLB definition: %v", err)
	}

	foundConfigFile := false
	for _, rawOp := range def.ToPB().Def {
		var op pb.Op
		if err := op.Unmarshal(rawOp); err != nil {
			t.Fatalf("unmarshal LLB operation: %v", err)
		}

		if execOp := op.GetExec(); execOp != nil {
			for _, arg := range execOp.Meta.Args {
				if strings.Contains(arg, marker) {
					t.Fatalf("finetune config content was interpolated into a shell command: %q", arg)
				}
			}
		}

		fileOp := op.GetFile()
		if fileOp == nil {
			continue
		}
		for _, action := range fileOp.Actions {
			mkfile := action.GetMkfile()
			if mkfile == nil || mkfile.Path != "/config.yaml" {
				continue
			}
			foundConfigFile = true
			if mkfile.Mode != 0o644 {
				t.Errorf("config file mode = %#o, want %#o", mkfile.Mode, 0o644)
			}
			if !bytes.Equal(mkfile.Data, wantConfig) {
				t.Errorf("config file contents = %q, want %q", mkfile.Data, wantConfig)
			}
		}
	}

	if !foundConfigFile {
		t.Fatal("LLB definition does not create /config.yaml with a file operation")
	}
}
