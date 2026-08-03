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

func TestAikit2LLBRequiresGPUCacheKey(t *testing.T) {
	_, err := Aikit2LLB(fineTuneTestConfig(), Options{})
	if err == nil || !strings.Contains(err.Error(), "GPU cache key") {
		t.Fatalf("Aikit2LLB() error = %v, want missing GPU cache key error", err)
	}
}

func TestAikit2LLBSeparatesTrainingAndExportPhases(t *testing.T) {
	cfg := fineTuneTestConfig()
	cfg.BaseModel = "model with \"quotes\"\nand $HOME `literal`"
	cfg.Datasets = []config.Dataset{{Source: "https://example.invalid/data?value=$x&other=`y`", Type: utils.DatasetAlpaca}}

	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, cfg))
	wantEnv := []string{
		"PATH=" + system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin",
		"LD_LIBRARY_PATH=/usr/local/cuda/lib64",
		"UV_CACHE_DIR=/root/.cache/uv",
		"UV_LINK_MODE=copy",
		"HF_HOME=/root/.cache/huggingface",
		"HF_DATASETS_CACHE=" + datasetsCachePath,
	}
	wantGPUEnv := append(slices.Clone(wantEnv), nvidiaCacheKey+"=session:test-session")
	for _, graphOp := range ops {
		if execOp := graphOp.op.GetExec(); execOp != nil && !slices.Equal(execOp.Meta.Env, wantEnv) && !slices.Equal(execOp.Meta.Env, wantGPUEnv) {
			t.Fatalf("exec environment = %#v, want base %#v or GPU %#v", execOp.Meta.Env, wantEnv, wantGPUEnv)
		}
	}

	dependencyOp := findFineTuneExec(t, ops, "uv pip sync")
	trainingOp := findFineTuneExec(t, ops, "target_unsloth.py train")
	exportOp := findFineTuneExec(t, ops, "target_unsloth.py export")
	if dependencyOp.index >= trainingOp.index || trainingOp.index >= exportOp.index {
		t.Fatalf("dependency, training, and export op indexes must be ordered: dependency=%d training=%d export=%d", dependencyOp.index, trainingOp.index, exportOp.index)
	}

	_, scriptFile := findFineTuneFile(t, ops, "/target_unsloth.py")
	if scriptFile.Mode != 0o755 || !slices.Equal(scriptFile.Data, finetunescript.TargetUnsloth) {
		t.Fatalf("script mkfile = mode %o data %q, want mode %o embedded script", scriptFile.Mode, string(scriptFile.Data), 0o755)
	}
	_, pylockFile := findFineTuneFile(t, ops, "/pylock.toml")
	if pylockFile.Mode != 0o644 || !slices.Equal(pylockFile.Data, finetunescript.UnslothPylock) {
		t.Fatalf("pylock mkfile = mode %o, want mode %o and embedded lock", pylockFile.Mode, 0o644)
	}
	_, bootstrapFile := findFineTuneFile(t, ops, "/uv-bootstrap.txt")
	if bootstrapFile.Mode != 0o644 || !slices.Equal(bootstrapFile.Data, finetunescript.UVBootstrap) {
		t.Fatalf("uv bootstrap mkfile = mode %o, want mode %o and embedded requirement", bootstrapFile.Mode, 0o644)
	}

	_, trainingConfigFile := findFineTuneFile(t, ops, "/train-config.yaml")
	wantTrainingConfig := mustMarshalYAML(unslothTrainingConfig{
		BaseModel: cfg.BaseModel,
		Datasets:  cfg.Datasets,
		Config:    cfg.Config,
	})
	if trainingConfigFile.Mode != 0o600 || !slices.Equal(trainingConfigFile.Data, wantTrainingConfig) {
		t.Fatalf("training config = mode %o data %q, want mode %o data %q", trainingConfigFile.Mode, string(trainingConfigFile.Data), 0o600, string(wantTrainingConfig))
	}

	_, exportConfigFile := findFineTuneFile(t, ops, "/export-config.yaml")
	wantExportConfig := unslothExportConfig{BaseModel: cfg.BaseModel, Config: cfg.Config}
	wantExportConfig.Output.Quantize = cfg.Output.Quantize
	wantExportData := mustMarshalYAML(wantExportConfig)
	if exportConfigFile.Mode != 0o600 || !slices.Equal(exportConfigFile.Data, wantExportData) {
		t.Fatalf("export config = mode %o data %q, want mode %o data %q", exportConfigFile.Mode, string(exportConfigFile.Data), 0o600, string(wantExportData))
	}

	assertReadonlyMount(t, trainingOp, "/aikit-bin")
	assertReadonlyMount(t, trainingOp, "/aikit-config")
	assertReadonlyMount(t, exportOp, "/aikit-bin")
	assertReadonlyMount(t, exportOp, "/aikit-config")

	outputCopy := findFineTuneGGUFCopy(t, ops)
	if strings.TrimPrefix(outputCopy.Src, "/") != "model/*.gguf" {
		t.Errorf("GGUF copy source = %q, want model/*.gguf", outputCopy.Src)
	}
	if strings.TrimPrefix(outputCopy.Dest, "/") != "output-q4_k_m.gguf" {
		t.Errorf("GGUF copy destination = %q, want output-q4_k_m.gguf", outputCopy.Dest)
	}
}

func TestAikit2LLBUsesFrozenIsolatedEnvironmentAndCaches(t *testing.T) {
	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, fineTuneTestConfig()))
	dependencyOp := findFineTuneExec(t, ops, "uv pip sync")
	dependencyCommand := strings.Join(dependencyOp.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		"--require-hashes",
		"/aikit-lock/uv-bootstrap.txt",
		"uv venv --python /usr/bin/python3 " + pythonVenv,
		"uv pip sync --preview-features pylock --require-hashes --python " + pythonVenv + "/bin/python /aikit-lock/pylock.toml",
	} {
		if !strings.Contains(dependencyCommand, fragment) {
			t.Errorf("dependency command does not contain %q: %q", fragment, dependencyCommand)
		}
	}
	for _, stale := range []string{"--system-site-packages", "pip install --upgrade", "git+https://github.com/unslothai/unsloth.git"} {
		if strings.Contains(dependencyCommand, stale) {
			t.Errorf("dependency command still contains non-reproducible fragment %q: %q", stale, dependencyCommand)
		}
	}
	for _, fragment := range []string{
		`name = "torch"`, `version = "2.10.0+cu126"`,
		`name = "unsloth"`, `version = "2026.8.1"`,
		`name = "unsloth-zoo"`, `name = "xformers"`,
	} {
		if !strings.Contains(string(finetunescript.UnslothPylock), fragment) {
			t.Errorf("embedded pylock does not contain %q", fragment)
		}
	}
	if !strings.Contains(string(finetunescript.UVBootstrap), "uv=="+uvVersion) || !strings.Contains(string(finetunescript.UVBootstrap), "--hash=sha256:") {
		t.Fatalf("uv bootstrap is not versioned and hashed: %q", finetunescript.UVBootstrap)
	}

	assertCacheMount(t, dependencyOp, "/root/.cache/uv", uvCacheID, pb.CacheSharingOpt_SHARED)
	trainingOp := findFineTuneExec(t, ops, "target_unsloth.py train")
	assertCacheMount(t, trainingOp, "/root/.cache/huggingface", huggingFaceCacheID, pb.CacheSharingOpt_SHARED)
	assertCacheMount(t, trainingOp, datasetsCachePath, datasetsCacheID, pb.CacheSharingOpt_SHARED)
	assertCacheMount(t, trainingOp, "/root/.cache/torch", torchCacheID, pb.CacheSharingOpt_SHARED)
	assertCacheMount(t, trainingOp, "/root/.triton", tritonCacheID, pb.CacheSharingOpt_SHARED)
	exportOp := findFineTuneExec(t, ops, "target_unsloth.py export")
	assertCacheMount(t, exportOp, "/root/.unsloth", llamaCacheID, pb.CacheSharingOpt_SHARED)
}

func TestAikit2LLBCacheBoundaries(t *testing.T) {
	base := fineTuneTestConfig()
	baseOps := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, base))
	baseTrain := findFineTuneExec(t, baseOps, "target_unsloth.py train")
	baseExport := findFineTuneExec(t, baseOps, "target_unsloth.py export")

	nameChanged := *base
	nameChanged.Output.Name = "renamed"
	nameOps := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, &nameChanged))
	if got := findFineTuneExec(t, nameOps, "target_unsloth.py train").digest; got != baseTrain.digest {
		t.Fatalf("output name change invalidated training: got %s, want %s", got, baseTrain.digest)
	}
	if got := findFineTuneExec(t, nameOps, "target_unsloth.py export").digest; got != baseExport.digest {
		t.Fatalf("output name change invalidated export: got %s, want %s", got, baseExport.digest)
	}
	if got := strings.TrimPrefix(findFineTuneGGUFCopy(t, nameOps).Dest, "/"); got != "renamed-q4_k_m.gguf" {
		t.Fatalf("renamed output destination = %q, want renamed-q4_k_m.gguf", got)
	}

	quantizeChanged := *base
	quantizeChanged.Output.Quantize = "q8_0"
	quantizeOps := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, &quantizeChanged))
	if got := findFineTuneExec(t, quantizeOps, "target_unsloth.py train").digest; got != baseTrain.digest {
		t.Fatalf("quantization change invalidated training: got %s, want %s", got, baseTrain.digest)
	}
	if got := findFineTuneExec(t, quantizeOps, "target_unsloth.py export").digest; got == baseExport.digest {
		t.Fatalf("quantization change did not invalidate export: %s", got)
	}

	trainingChanged := *base
	trainingChanged.BaseModel = "different-model"
	trainingOps := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, &trainingChanged))
	if got := findFineTuneExec(t, trainingOps, "target_unsloth.py train").digest; got == baseTrain.digest {
		t.Fatalf("training config change did not invalidate training: %s", got)
	}
	if got := findFineTuneExec(t, trainingOps, "target_unsloth.py export").digest; got == baseExport.digest {
		t.Fatalf("training config change did not invalidate export: %s", got)
	}
}

func TestAikit2LLBUsesNvidiaCDIWithoutInsecureSecurity(t *testing.T) {
	definition := marshalFineTuneDefinition(t, fineTuneTestConfig())
	ops := decodeFineTuneDefinition(t, definition)
	trainingOp := findFineTuneExec(t, ops, "target_unsloth.py train")
	exportOp := findFineTuneExec(t, ops, "target_unsloth.py export")
	dependencyOp := findFineTuneExec(t, ops, "uv pip sync")

	if devices := dependencyOp.op.GetExec().GetCdiDevices(); len(devices) != 0 {
		t.Fatalf("dependency installation requested CDI devices: %#v", devices)
	}

	for _, phase := range []struct {
		name string
		op   fineTuneDefinitionOp
	}{
		{name: "training", op: trainingOp},
		{name: "export", op: exportOp},
	} {
		t.Run(phase.name, func(t *testing.T) {
			execOp := phase.op.op.GetExec()
			devices := execOp.GetCdiDevices()
			if len(devices) != 1 || devices[0].Name != nvidiaCDIDevice || devices[0].Optional {
				t.Fatalf("CDI devices = %#v, want required %q", devices, nvidiaCDIDevice)
			}
			if execOp.Security != pb.SecurityMode_SANDBOX {
				t.Fatalf("security mode = %s, want sandbox", execOp.Security)
			}
			if command := strings.Join(execOp.Meta.Args, "\x00"); !strings.Contains(command, "nvidia-smi") {
				t.Fatalf("phase command does not verify CDI GPU access: %q", command)
			}
			metadata, ok := definition.Metadata[phase.op.digest]
			if !ok {
				t.Fatalf("metadata for operation %s not found", phase.op.digest)
			}
			if !metadata.Caps[pb.CapExecMetaCDI] {
				t.Fatal("exec.meta.cdi capability is not set")
			}
		})
	}

	for _, graphOp := range ops {
		execOp := graphOp.op.GetExec()
		if execOp == nil {
			continue
		}
		command := strings.Join(execOp.Meta.Args, "\x00")
		for _, stale := range []string{"/proc/devices", "mknod", "NVIDIA-Linux-x86_64", "/proc/driver/nvidia/version"} {
			if strings.Contains(command, stale) {
				t.Errorf("exec command still contains manual NVIDIA setup fragment %q: %q", stale, command)
			}
		}
	}
}

func TestAikit2LLBUsesDriverVersionOnlyAsGPUCacheKey(t *testing.T) {
	baseDefinition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{NVIDIADriverVersion: "590.48.01", BuildSessionID: "session-a"})
	baseOps := decodeFineTuneDefinition(t, baseDefinition)
	baseDependency := findFineTuneExec(t, baseOps, "uv pip sync")
	baseTraining := findFineTuneExec(t, baseOps, "target_unsloth.py train")
	baseExport := findFineTuneExec(t, baseOps, "target_unsloth.py export")

	if slices.Contains(baseDependency.op.GetExec().Meta.Env, nvidiaCacheKey+"=driver:590.48.01") {
		t.Fatal("driver cache key invalidated dependency installation")
	}
	for _, phase := range []fineTuneDefinitionOp{baseTraining, baseExport} {
		if !slices.Contains(phase.op.GetExec().Meta.Env, nvidiaCacheKey+"=driver:590.48.01") {
			t.Fatalf("GPU phase environment does not contain driver cache key: %#v", phase.op.GetExec().Meta.Env)
		}
		if baseDefinition.Metadata[phase.digest].IgnoreCache {
			t.Fatal("GPU phase ignored cache despite an explicit driver version")
		}
	}

	changedDefinition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{NVIDIADriverVersion: "590.50.02", BuildSessionID: "session-b"})
	changedOps := decodeFineTuneDefinition(t, changedDefinition)
	if got := findFineTuneExec(t, changedOps, "uv pip sync").digest; got != baseDependency.digest {
		t.Fatalf("driver version change invalidated dependency installation: got %s, want %s", got, baseDependency.digest)
	}
	if got := findFineTuneExec(t, changedOps, "target_unsloth.py train").digest; got == baseTraining.digest {
		t.Fatalf("driver version change did not invalidate training: %s", got)
	}
	if got := findFineTuneExec(t, changedOps, "target_unsloth.py export").digest; got == baseExport.digest {
		t.Fatalf("driver version change did not invalidate export: %s", got)
	}
}

func TestAikit2LLBUsesSessionCacheKeyWithoutDriverVersion(t *testing.T) {
	baseDefinition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{BuildSessionID: "session-a"})
	baseOps := decodeFineTuneDefinition(t, baseDefinition)
	baseDependency := findFineTuneExec(t, baseOps, "uv pip sync")
	baseTraining := findFineTuneExec(t, baseOps, "target_unsloth.py train")
	baseExport := findFineTuneExec(t, baseOps, "target_unsloth.py export")

	if slices.Contains(baseDependency.op.GetExec().Meta.Env, nvidiaCacheKey+"=session:session-a") {
		t.Fatal("session cache key invalidated dependency installation")
	}
	for _, phase := range []fineTuneDefinitionOp{baseTraining, baseExport} {
		if !slices.Contains(phase.op.GetExec().Meta.Env, nvidiaCacheKey+"=session:session-a") {
			t.Fatalf("GPU phase environment does not contain session cache key: %#v", phase.op.GetExec().Meta.Env)
		}
		if baseDefinition.Metadata[phase.digest].IgnoreCache {
			t.Fatal("GPU phase prunes persistent caches when using a session cache key")
		}
	}

	changedDefinition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{BuildSessionID: "session-b"})
	changedOps := decodeFineTuneDefinition(t, changedDefinition)
	if got := findFineTuneExec(t, changedOps, "uv pip sync").digest; got != baseDependency.digest {
		t.Fatalf("session change invalidated dependency installation: got %s, want %s", got, baseDependency.digest)
	}
	if got := findFineTuneExec(t, changedOps, "target_unsloth.py train").digest; got == baseTraining.digest {
		t.Fatalf("session change did not invalidate training: %s", got)
	}
	if got := findFineTuneExec(t, changedOps, "target_unsloth.py export").digest; got == baseExport.digest {
		t.Fatalf("session change did not invalidate export: %s", got)
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
	return marshalFineTuneDefinitionWithOptions(t, cfg, Options{BuildSessionID: "test-session"})
}

func marshalFineTuneDefinitionWithOptions(t *testing.T, cfg *config.FineTuneConfig, options Options) *llb.Definition {
	t.Helper()

	state, err := Aikit2LLB(cfg, options)
	if err != nil {
		t.Fatalf("convert fine-tune config: %v", err)
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

func findFineTuneMount(t *testing.T, graphOp fineTuneDefinitionOp, destination string) *pb.Mount {
	t.Helper()

	execOp := graphOp.op.GetExec()
	if execOp == nil {
		t.Fatalf("op %s is not an exec op", graphOp.digest)
	}
	for _, mount := range execOp.Mounts {
		if mount.Dest == destination {
			return mount
		}
	}
	t.Fatalf("mount %q not found in exec op %s", destination, graphOp.digest)
	return nil
}

func assertReadonlyMount(t *testing.T, graphOp fineTuneDefinitionOp, destination string) {
	t.Helper()

	mount := findFineTuneMount(t, graphOp, destination)
	if !mount.Readonly || mount.MountType != pb.MountType_BIND {
		t.Fatalf("mount %q = readonly %t type %s, want readonly bind mount", destination, mount.Readonly, mount.MountType)
	}
}

func assertCacheMount(t *testing.T, graphOp fineTuneDefinitionOp, destination, id string, sharing pb.CacheSharingOpt) {
	t.Helper()

	mount := findFineTuneMount(t, graphOp, destination)
	if mount.MountType != pb.MountType_CACHE || mount.CacheOpt == nil {
		t.Fatalf("mount %q = type %s cacheOpt %#v, want cache mount", destination, mount.MountType, mount.CacheOpt)
	}
	if mount.CacheOpt.ID != id || mount.CacheOpt.Sharing != sharing {
		t.Fatalf("cache mount %q = id %q sharing %s, want id %q sharing %s", destination, mount.CacheOpt.ID, mount.CacheOpt.Sharing, id, sharing)
	}
}

func cloneFineTuneDefinition(definition [][]byte) [][]byte {
	cloned := make([][]byte, len(definition))
	for i, data := range definition {
		cloned[i] = slices.Clone(data)
	}
	return cloned
}
