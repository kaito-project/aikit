package finetune

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	"github.com/moby/buildkit/util/system"
	digest "github.com/opencontainers/go-digest"
	"gopkg.in/yaml.v2"
)

type fineTuneDefinitionOp struct {
	digest digest.Digest
	index  int
	op     *pb.Op
}

func TestAikit2LLBDefinitionIsDeterministic(t *testing.T) {
	cfg := fineTuneTestConfig()

	var wantDefinition [][]byte
	var wantHead digest.Digest
	for i := 0; i < 100; i++ {
		definition := marshalFineTuneDefinition(t, cfg)
		head, err := definition.Head()
		if err != nil {
			t.Fatalf("resolve definition head: %v", err)
		}

		if i == 0 {
			wantDefinition = cloneFineTuneDefinition(definition.Def)
			wantHead = head
			continue
		}
		if head != wantHead {
			t.Fatalf("definition head changed on conversion %d: got %s, want %s", i, head, wantHead)
		}
		if !reflect.DeepEqual(definition.Def, wantDefinition) {
			t.Fatalf("definition bytes changed on conversion %d", i)
		}
	}
}

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

	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, cfg))
	for _, graphOp := range ops {
		if execOp := graphOp.op.GetExec(); execOp != nil {
			for _, arg := range execOp.Meta.Args {
				if strings.Contains(arg, marker) {
					t.Fatalf("finetune config content was interpolated into a shell command: %q", arg)
				}
			}
		}
	}

	_, configFile := findFineTuneConfigFile(t, ops)
	if configFile.Mode != 0o644 {
		t.Errorf("config file mode = %#o, want %#o", configFile.Mode, 0o644)
	}
	if !slices.Equal(configFile.Data, wantConfig) {
		t.Errorf("config file contents = %q, want %q", configFile.Data, wantConfig)
	}
}

func TestAikit2LLBMaterializesConfigAfterDependencies(t *testing.T) {
	cfg := fineTuneTestConfig()
	cfg.BaseModel = "model with \"quotes\"\nand $HOME `literal`"
	cfg.Datasets = []config.Dataset{{Source: "https://example.invalid/data?value=$x&other=`y`", Type: utils.DatasetAlpaca}}

	definition := marshalFineTuneDefinition(t, cfg)
	ops := decodeFineTuneDefinition(t, definition)

	wantEnv := []string{
		"PATH=" + system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_VISIBLE_DEVICES=all",
		"LD_LIBRARY_PATH=/usr/local/cuda/lib64",
	}
	for _, graphOp := range ops {
		if exec := graphOp.op.GetExec(); exec != nil && !slices.Equal(exec.Meta.Env, wantEnv) {
			t.Fatalf("exec environment = %#v, want %#v", exec.Meta.Env, wantEnv)
		}
	}

	dependencyOp := findFineTuneExec(t, ops, "uv pip install --upgrade --force-reinstall")
	trainingOp := findFineTuneExec(t, ops, "python -m target_unsloth")
	configOp, configFile := findFineTuneConfigFile(t, ops)
	wantConfig, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal expected config: %v", err)
	}
	if configFile.Mode != 0o644 || !slices.Equal(configFile.Data, wantConfig) {
		t.Fatalf("config mkfile = mode %o data %q, want mode %o data %q", configFile.Mode, string(configFile.Data), 0o644, string(wantConfig))
	}
	if dependencyOp.index >= configOp.index {
		t.Fatalf("dependency op index %d must precede config op index %d", dependencyOp.index, configOp.index)
	}
	if len(trainingOp.op.Inputs) != 1 || trainingOp.op.Inputs[0].Digest != configOp.digest.String() {
		t.Fatalf("training op inputs = %#v, want config digest %s", trainingOp.op.Inputs, configOp.digest)
	}
	for _, graphOp := range ops {
		if exec := graphOp.op.GetExec(); exec != nil {
			command := strings.Join(exec.Meta.Args, "\x00")
			if strings.Contains(command, "echo -n") && strings.Contains(command, "/config.yaml") {
				t.Fatalf("config unexpectedly materialized by shell command %q", command)
			}
		}
	}

	changedConfig := *cfg
	changedConfig.BaseModel = "a config-only change"
	changedOps := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, &changedConfig))
	changedDependencyOp := findFineTuneExec(t, changedOps, "uv pip install --upgrade --force-reinstall")
	changedConfigOp, _ := findFineTuneConfigFile(t, changedOps)
	if dependencyOp.digest != changedDependencyOp.digest {
		t.Fatalf("config-only change invalidated dependency op: got %s, want %s", changedDependencyOp.digest, dependencyOp.digest)
	}
	if configOp.digest == changedConfigOp.digest {
		t.Fatalf("config file op digest did not change after config change: %s", configOp.digest)
	}
}

func fineTuneTestConfig() *config.FineTuneConfig {
	return &config.FineTuneConfig{
		APIVersion: utils.APIv1alpha1,
		Target:     utils.TargetUnsloth,
		BaseModel:  "base-model",
		Config: config.FineTuneConfigSpec{
			Unsloth: config.FineTuneConfigUnslothSpec{
				MaxSeqLength:              2048,
				LoadIn4bit:                true,
				BatchSize:                 2,
				GradientAccumulationSteps: 4,
				MaxSteps:                  20,
				LearningRate:              0.0002,
			},
		},
		Output: config.FineTuneOutputSpec{Name: "output", Quantize: "q4_k_m"},
	}
}

func marshalFineTuneDefinition(t *testing.T, cfg *config.FineTuneConfig) *llb.Definition {
	t.Helper()

	state, err := Aikit2LLB(cfg)
	if err != nil {
		t.Fatalf("convert fine-tune config to LLB: %v", err)
	}
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal fine-tune definition: %v", err)
	}
	return definition
}

func decodeFineTuneDefinition(t *testing.T, definition *llb.Definition) []fineTuneDefinitionOp {
	t.Helper()

	ops := make([]fineTuneDefinitionOp, 0, len(definition.Def))
	for i, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		ops = append(ops, fineTuneDefinitionOp{digest: digest.FromBytes(data), index: i, op: op})
	}
	return ops
}

func findFineTuneExec(t *testing.T, ops []fineTuneDefinitionOp, commandFragment string) fineTuneDefinitionOp {
	t.Helper()

	for _, graphOp := range ops {
		if exec := graphOp.op.GetExec(); exec != nil && strings.Contains(strings.Join(exec.Meta.Args, "\x00"), commandFragment) {
			return graphOp
		}
	}
	t.Fatalf("exec op containing %q not found", commandFragment)
	return fineTuneDefinitionOp{}
}

func findFineTuneConfigFile(t *testing.T, ops []fineTuneDefinitionOp) (fineTuneDefinitionOp, *pb.FileActionMkFile) {
	t.Helper()

	for _, graphOp := range ops {
		if fileOp := graphOp.op.GetFile(); fileOp != nil {
			for _, action := range fileOp.Actions {
				if mkfile := action.GetMkfile(); mkfile != nil && mkfile.Path == "/config.yaml" {
					return graphOp, mkfile
				}
			}
		}
	}
	t.Fatal("config mkfile action not found")
	return fineTuneDefinitionOp{}, nil
}

func cloneFineTuneDefinition(definition [][]byte) [][]byte {
	cloned := make([][]byte, len(definition))
	for i, data := range definition {
		cloned[i] = slices.Clone(data)
	}
	return cloned
}
