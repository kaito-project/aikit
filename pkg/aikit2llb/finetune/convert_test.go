package finetune

import (
	"context"
	"os/exec"
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

func TestAikit2LLBSeparatesTrainingAndExportPhases(t *testing.T) {
	cfg := fineTuneTestConfig()
	cfg.BaseModel = "model with \"quotes\"\nand $HOME `literal`"
	cfg.Datasets = []config.Dataset{{Source: "https://example.invalid/data?value=$x&other=`y`", Type: utils.DatasetAlpaca}}

	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, cfg))
	wantEnv := []string{
		"PATH=" + system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.6",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_VISIBLE_DEVICES=all",
		"LD_LIBRARY_PATH=/usr/local/cuda/lib64",
		"UV_CACHE_DIR=/root/.cache/uv",
		"UV_LINK_MODE=copy",
		"HF_HOME=/root/.cache/huggingface",
		"HF_DATASETS_CACHE=" + datasetsCachePath,
	}
	for _, graphOp := range ops {
		if execOp := graphOp.op.GetExec(); execOp != nil && !slices.Equal(execOp.Meta.Env, wantEnv) {
			t.Fatalf("exec environment = %#v, want %#v", execOp.Meta.Env, wantEnv)
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

func TestAikit2LLBDiscoversNvidiaDeviceMajors(t *testing.T) {
	ops := decodeFineTuneDefinition(t, marshalFineTuneDefinition(t, fineTuneTestConfig()))
	trainingOp := findFineTuneExec(t, ops, "target_unsloth.py train")
	trainingCommand := strings.Join(trainingOp.op.GetExec().Meta.Args, "\x00")

	for _, fragment := range []string{"/proc/devices", "nvidia-frontend", "nvidia-uvm", "$NVIDIA_UVM_MAJOR", "$NVIDIA_MAJOR"} {
		if !strings.Contains(trainingCommand, fragment) {
			t.Errorf("training command does not contain dynamic NVIDIA device fragment %q: %q", fragment, trainingCommand)
		}
	}
	if strings.Contains(trainingCommand, "nvidia-uvm c 235") {
		t.Fatalf("training command still contains the stale hard-coded NVIDIA UVM major: %q", trainingCommand)
	}
}

func TestAikit2LLBUsesExplicitNvidiaDriverVersion(t *testing.T) {
	definition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{NVIDIADriverVersion: "590.48.01"})
	ops := decodeFineTuneDefinition(t, definition)
	driverOp := findFineTuneExec(t, ops, "NVIDIA-Linux-x86_64-$VERSION.run")
	driverCommand := strings.Join(driverOp.op.GetExec().Meta.Args, "\x00")

	if !strings.Contains(driverCommand, "VERSION='590.48.01'") {
		t.Fatalf("driver command does not contain explicit version: %q", driverCommand)
	}
	if strings.Contains(driverCommand, `VERSION=$(sed -n`) {
		t.Fatalf("explicit driver command still reads host version: %q", driverCommand)
	}
}

func TestAikit2LLBAutoDriverVersionHasActionableFailure(t *testing.T) {
	definition := marshalFineTuneDefinition(t, fineTuneTestConfig())
	ops := decodeFineTuneDefinition(t, definition)
	driverOp := findFineTuneExec(t, ops, "NVIDIA-Linux-x86_64-$VERSION.run")
	driverCommand := strings.Join(driverOp.op.GetExec().Meta.Args, "\x00")

	for _, fragment := range []string{
		`/proc/driver/nvidia/version 2>/dev/null || true)`,
		`if [ -z "$VERSION" ]; then`,
		`failed to resolve NVIDIA driver version from /proc/driver/nvidia/version`,
		`--build-arg nvidiaDriverVersion=<major.minor.patch>`,
	} {
		if !strings.Contains(driverCommand, fragment) {
			t.Errorf("driver command does not contain actionable failure fragment %q: %q", fragment, driverCommand)
		}
	}
	if strings.Contains(driverCommand, `test -n "$VERSION"`) {
		t.Fatalf("driver command still uses a silent version check: %q", driverCommand)
	}
}

func TestAikit2LLBCachesNvidiaInstallerAtomically(t *testing.T) {
	definition := marshalFineTuneDefinitionWithOptions(t, fineTuneTestConfig(), Options{NVIDIADriverVersion: "590.48.01"})
	ops := decodeFineTuneDefinition(t, definition)
	driverOp := findFineTuneExec(t, ops, "NVIDIA-Linux-x86_64-$VERSION.run")
	driverCommand := strings.Join(driverOp.op.GetExec().Meta.Args, "\x00")

	for _, fragment := range []string{
		"CHECKSUM_URL=$DOWNLOAD_URL.sha256sum",
		`DOWNLOAD_DIR=$(mktemp -d "$DRIVER_CACHE/.${INSTALLER_NAME}.download.XXXXXX")`,
		`INSTALLER_TMP=$DOWNLOAD_DIR/$INSTALLER_NAME`,
		`CHECKSUM_TMP=$DOWNLOAD_DIR/$INSTALLER_NAME.sha256sum`,
		`trap 'rm -rf "$DOWNLOAD_DIR"' 0 HUP INT TERM`,
		`wget --no-verbose "$CHECKSUM_URL" -O "$CHECKSUM_TMP"`,
		`(cd "$DRIVER_CACHE" && sha256sum --check --status "$CHECKSUM_TMP")`,
		`wget --no-verbose "$DOWNLOAD_URL" -O "$INSTALLER_TMP"`,
		`(cd "$DOWNLOAD_DIR" && sha256sum --check --status "$CHECKSUM_TMP")`,
		`mv "$INSTALLER_TMP" "$INSTALLER"`,
		`rm -rf "$DOWNLOAD_DIR"`,
	} {
		if !strings.Contains(driverCommand, fragment) {
			t.Errorf("driver command does not contain atomic cache fragment %q: %q", fragment, driverCommand)
		}
	}
	if strings.Contains(driverCommand, `-O "$INSTALLER"`) {
		t.Fatalf("driver command downloads directly to the final cache path: %q", driverCommand)
	}
	downloadIndex := strings.Index(driverCommand, `wget --no-verbose "$DOWNLOAD_URL" -O "$INSTALLER_TMP"`)
	verifyIndex := strings.LastIndex(driverCommand, `(cd "$DOWNLOAD_DIR" && sha256sum --check --status "$CHECKSUM_TMP")`)
	renameIndex := strings.Index(driverCommand, `mv "$INSTALLER_TMP" "$INSTALLER"`)
	if downloadIndex == -1 || verifyIndex <= downloadIndex || renameIndex <= verifyIndex {
		t.Fatalf("driver installer must be downloaded to a temporary path, verified, then atomically renamed: %q", driverCommand)
	}
}

func TestNvidiaPrimaryMajorAWK(t *testing.T) {
	tests := []struct {
		name        string
		devices     string
		wantMajor   string
		wantFailure bool
	}{
		{
			name:      "frontend",
			devices:   "Character devices:\n195 nvidia-frontend\n511 nvidia-uvm\n",
			wantMajor: "195",
		},
		{
			name:      "legacy fallback",
			devices:   "Character devices:\n195 nvidia\n511 nvidia-uvm\n",
			wantMajor: "195",
		},
		{
			name:      "frontend preferred over legacy",
			devices:   "Character devices:\n195 nvidia\n509 nvidia-frontend\n511 nvidia-uvm\n",
			wantMajor: "509",
		},
		{
			name:        "missing primary device",
			devices:     "Character devices:\n511 nvidia-uvm\n",
			wantFailure: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("awk", nvidiaPrimaryMajorAWK)
			cmd.Stdin = strings.NewReader(tt.devices)
			output, err := cmd.CombinedOutput()
			if tt.wantFailure {
				if err == nil {
					t.Fatalf("device discovery unexpectedly succeeded: %q", output)
				}
				if !strings.Contains(string(output), "expected nvidia-frontend or nvidia") {
					t.Fatalf("device discovery failure = %q, want actionable error", output)
				}
				return
			}

			if err != nil {
				t.Fatalf("device discovery failed: %v: %s", err, output)
			}
			if got := strings.TrimSpace(string(output)); got != tt.wantMajor {
				t.Fatalf("device major = %q, want %q", got, tt.wantMajor)
			}
		})
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
	return marshalFineTuneDefinitionWithOptions(t, cfg, Options{})
}

func marshalFineTuneDefinitionWithOptions(t *testing.T, cfg *config.FineTuneConfig, options Options) *llb.Definition {
	t.Helper()

	definition, err := Aikit2LLB(cfg, options).Marshal(context.Background())
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
