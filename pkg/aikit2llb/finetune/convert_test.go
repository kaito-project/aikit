package finetune

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	finetunescript "github.com/kaito-project/aikit/pkg/finetune"
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

func TestAikit2LLBMaterializesConfigAfterDependencies(t *testing.T) {
	cfg := fineTuneTestConfig()
	cfg.BaseModel = "model with \"quotes\"\nand $HOME `literal`"
	cfg.Datasets = []config.Dataset{{Source: "https://example.invalid/data?value=$x&other=`y`", Type: utils.DatasetAlpaca}}

	definition := marshalFineTuneDefinition(t, cfg)
	ops := decodeFineTuneDefinition(t, definition)

	wantEnv := []string{
		"PATH=" + system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.6",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_VISIBLE_DEVICES=all",
		"LD_LIBRARY_PATH=/usr/local/cuda/lib64",
	}
	for _, graphOp := range ops {
		if exec := graphOp.op.GetExec(); exec != nil && !slices.Equal(exec.Meta.Env, wantEnv) {
			t.Fatalf("exec environment = %#v, want %#v", exec.Meta.Env, wantEnv)
		}
	}

	dependencyOp := findFineTuneExec(t, ops, "unsloth[cu126-torch2100]")
	scriptOp, scriptFile := findFineTuneFile(t, ops, "/target_unsloth.py")
	trainingOp := findFineTuneExec(t, ops, "python -m target_unsloth")
	configOp, configFile := findFineTuneConfigFile(t, ops)
	if scriptFile.Mode != 0o755 || !slices.Equal(scriptFile.Data, finetunescript.TargetUnsloth) {
		t.Fatalf("script mkfile = mode %o data %q, want mode %o embedded script", scriptFile.Mode, string(scriptFile.Data), 0o755)
	}
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
	if dependencyOp.index >= scriptOp.index || scriptOp.index >= configOp.index {
		t.Fatalf("dependency, script, and config op indexes must be ordered: dependency=%d script=%d config=%d", dependencyOp.index, scriptOp.index, configOp.index)
	}
	if len(trainingOp.op.Inputs) != 1 || trainingOp.op.Inputs[0].Digest != configOp.digest.String() {
		t.Fatalf("training op inputs = %#v, want config digest %s", trainingOp.op.Inputs, configOp.digest)
	}
	outputCopy := findFineTuneGGUFCopy(t, ops)
	if strings.TrimPrefix(outputCopy.Src, "/") != "model/*.gguf" {
		t.Errorf("GGUF copy source = %q, want model/*.gguf", outputCopy.Src)
	}
	if strings.TrimPrefix(outputCopy.Dest, "/") != "output-q4_k_m.gguf" {
		t.Errorf("GGUF copy destination = %q, want output-q4_k_m.gguf", outputCopy.Dest)
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
	changedDependencyOp := findFineTuneExec(t, changedOps, "unsloth[cu126-torch2100]")
	changedScriptOp, _ := findFineTuneFile(t, changedOps, "/target_unsloth.py")
	changedConfigOp, _ := findFineTuneConfigFile(t, changedOps)
	if dependencyOp.digest != changedDependencyOp.digest {
		t.Fatalf("config-only change invalidated dependency op: got %s, want %s", changedDependencyOp.digest, dependencyOp.digest)
	}
	if configOp.digest == changedConfigOp.digest {
		t.Fatalf("config file op digest did not change after config change: %s", configOp.digest)
	}
	if scriptOp.digest != changedScriptOp.digest {
		t.Fatalf("config-only change invalidated script op: got %s, want %s", changedScriptOp.digest, scriptOp.digest)
	}
}

func TestAikit2LLBUsesCurrentUnslothDependencies(t *testing.T) {
	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, fineTuneTestConfig()))
	dependencyOp := findFineTuneExec(t, ops, "unsloth[cu126-torch2100]")
	dependencyCommand := strings.Join(dependencyOp.op.GetExec().Meta.Args, "\x00")

	wantFragments := []string{
		"torch==" + torchVersion,
		"unsloth[cu126-torch2100]==" + unslothVersion,
		"unsloth-zoo==" + unslothVersion,
		"--torch-backend=cu126",
	}
	for _, fragment := range wantFragments {
		if !strings.Contains(dependencyCommand, fragment) {
			t.Errorf("dependency command does not contain %q: %q", fragment, dependencyCommand)
		}
	}

	staleFragments := []string{
		"transformers==4.44.2",
		"torch==2.4.0",
		"torch==2.4.1",
		"git+https://github.com/unslothai/unsloth.git",
		"fb77505f8429566f5d21d6ea5318c342e8a67991",
	}
	for _, fragment := range staleFragments {
		if strings.Contains(dependencyCommand, fragment) {
			t.Errorf("dependency command still contains stale fragment %q: %q", fragment, dependencyCommand)
		}
	}
}

func TestAikit2LLBDiscoversNvidiaDeviceMajors(t *testing.T) {
	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, fineTuneTestConfig()))
	trainingOp := findFineTuneExec(t, ops, "python -m target_unsloth")
	trainingCommand := strings.Join(trainingOp.op.GetExec().Meta.Args, "\x00")

	for _, fragment := range []string{"/proc/devices", "nvidia-uvm", "$NVIDIA_UVM_MAJOR", "$NVIDIA_MAJOR"} {
		if !strings.Contains(trainingCommand, fragment) {
			t.Errorf("training command does not contain dynamic NVIDIA device fragment %q: %q", fragment, trainingCommand)
		}
	}
	if strings.Contains(trainingCommand, "nvidia-uvm c 235") {
		t.Fatalf("training command still contains the stale hard-coded NVIDIA UVM major: %q", trainingCommand)
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

	definition, err := Aikit2LLB(cfg).Marshal(context.Background())
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

	return findFineTuneFile(t, ops, "/config.yaml")
}

func findFineTuneFile(t *testing.T, ops []fineTuneDefinitionOp, path string) (fineTuneDefinitionOp, *pb.FileActionMkFile) {
	t.Helper()

	for _, graphOp := range ops {
		if fileOp := graphOp.op.GetFile(); fileOp != nil {
			for _, action := range fileOp.Actions {
				if mkfile := action.GetMkfile(); mkfile != nil && mkfile.Path == path {
					return graphOp, mkfile
				}
			}
		}
	}
	t.Fatalf("mkfile action for %q not found", path)
	return fineTuneDefinitionOp{}, nil
}

func findFineTuneGGUFCopy(t *testing.T, ops []fineTuneDefinitionOp) *pb.FileActionCopy {
	t.Helper()

	for _, graphOp := range ops {
		if fileOp := graphOp.op.GetFile(); fileOp != nil {
			for _, action := range fileOp.Actions {
				if copyAction := action.GetCopy(); copyAction != nil && strings.HasSuffix(copyAction.Src, "*.gguf") {
					return copyAction
				}
			}
		}
	}
	t.Fatal("GGUF copy action not found")
	return nil
}

func cloneFineTuneDefinition(definition [][]byte) [][]byte {
	cloned := make([][]byte, len(definition))
	for i, data := range definition {
		cloned[i] = slices.Clone(data)
	}
	return cloned
}
