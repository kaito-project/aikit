package finetune

import (
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	finetunescript "github.com/kaito-project/aikit/pkg/finetune"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/util/system"
	"github.com/pkg/errors"
	"gopkg.in/yaml.v2"
)

const (
	unslothVersion  = "2026.8.1"
	torchVersion    = "2.10.0"
	uvVersion       = "0.12.1"
	pythonVenv      = "/opt/aikit-venv"
	sourceVenv      = ". " + pythonVenv + "/bin/activate"
	nvidiaCDIDevice = "nvidia.com/gpu=0"
	nvidiaCacheKey  = "AIKIT_NVIDIA_CACHE_KEY"

	aptCacheID         = "aikit-finetune-apt-v1"
	aptListsCacheID    = "aikit-finetune-apt-lists-v1"
	huggingFaceCacheID = "aikit-unsloth-huggingface-v1"
	datasetsCacheID    = "aikit-unsloth-datasets-v1"
	torchCacheID       = "aikit-unsloth-torch-v1"
	tritonCacheID      = "aikit-unsloth-triton-v1"
	llamaCacheID       = "aikit-unsloth-llama-v1"
	uvCacheID          = "aikit-unsloth-uv-" + uvVersion + "-py310-cu126-torch-" + torchVersion + "-unsloth-" + unslothVersion
	datasetsCachePath  = "/tmp/aikit-datasets-cache"
)

type unslothTrainingConfig struct {
	BaseModel string                    `yaml:"baseModel"`
	Datasets  []config.Dataset          `yaml:"datasets"`
	Config    config.FineTuneConfigSpec `yaml:"config"`
}

type unslothExportConfig struct {
	BaseModel string                    `yaml:"baseModel"`
	Config    config.FineTuneConfigSpec `yaml:"config"`
	Output    struct {
		Quantize string `yaml:"quantize"`
	} `yaml:"output"`
}

// Options configures host-specific fine-tuning cache behavior.
type Options struct {
	NVIDIADriverVersion string
	BuildSessionID      string
	CDIDevice           string
}

func Aikit2LLB(c *config.FineTuneConfig, opts Options) (llb.State, error) {
	cacheKey := gpuCacheKey(opts)
	if cacheKey == "" {
		return llb.State{}, errors.New("GPU cache key requires an NVIDIA driver version or BuildKit session ID")
	}
	cdiDevice := opts.CDIDevice
	if cdiDevice == "" {
		cdiDevice = nvidiaCDIDevice
	}

	env := []struct {
		key   string
		value string
	}{
		{key: "PATH", value: system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin"},
		{key: "LD_LIBRARY_PATH", value: "/usr/local/cuda/lib64"},
		{key: "UV_CACHE_DIR", value: "/root/.cache/uv"},
		{key: "UV_LINK_MODE", value: "copy"},
		{key: "HF_HOME", value: "/root/.cache/huggingface"},
		{key: "HF_DATASETS_CACHE", value: datasetsCachePath},
	}

	state := llb.Image(utils.CudaDevel)
	for _, entry := range env {
		state = state.AddEnv(entry.key, entry.value)
	}

	// Keep OS package installation cacheable across fine-tuning builds.
	state = state.Run(
		utils.Sh("rm -f /etc/apt/apt.conf.d/docker-clean && apt-get update && apt-get install -y --no-install-recommends cmake git python-is-python3 python3 python3-dev python3-pip"),
		persistentCacheMount("/var/cache/apt", aptCacheID, llb.CacheMountLocked),
		persistentCacheMount("/var/lib/apt/lists", aptListsCacheID, llb.CacheMountLocked),
	).Root()

	if c.Target != utils.TargetUnsloth {
		return llb.Scratch(), nil
	}

	lockState := llb.Scratch().
		File(llb.Mkfile("/pylock.toml", 0o644, finetunescript.UnslothPylock)).
		File(llb.Mkfile("/uv-bootstrap.txt", 0o644, finetunescript.UVBootstrap))
	dependencyCommand := "python3 -m pip install --disable-pip-version-check --no-deps --only-binary=:all: " +
		"--require-hashes -r /aikit-lock/uv-bootstrap.txt && " +
		"uv venv --python /usr/bin/python3 %[1]s && " +
		"uv pip sync --preview-features pylock --require-hashes --python %[1]s/bin/python /aikit-lock/pylock.toml"
	state = state.Run(
		utils.Shf(dependencyCommand, pythonVenv),
		llb.AddMount("/aikit-lock", lockState, llb.Readonly),
		persistentCacheMount("/root/.cache/uv", uvCacheID, llb.CacheMountShared),
	).Root()

	// CDI injects the host driver at execution time. Keep host identity in
	// the GPU phase cache key without invalidating OS or Python dependencies.
	state = state.AddEnv(nvidiaCacheKey, cacheKey)

	scriptState := llb.Scratch().File(llb.Mkfile("/target_unsloth.py", 0o755, finetunescript.TargetUnsloth))
	trainingConfig := unslothTrainingConfig{
		BaseModel: c.BaseModel,
		Datasets:  c.Datasets,
		Config:    c.Config,
	}
	trainingConfigState := llb.Scratch().File(llb.Mkfile("/train-config.yaml", 0o600, mustMarshalYAML(trainingConfig)))
	state = runUnslothPhase(state, scriptState, trainingConfigState, "train", cdiDevice, false)

	exportConfig := unslothExportConfig{
		BaseModel: c.BaseModel,
		Config:    c.Config,
	}
	exportConfig.Output.Quantize = c.Output.Quantize
	exportConfigState := llb.Scratch().File(llb.Mkfile("/export-config.yaml", 0o600, mustMarshalYAML(exportConfig)))
	state = runUnslothPhase(state, scriptState, exportConfigState, "export", cdiDevice, true)

	const inputFile = "model/*.gguf"
	copyOpts := []llb.CopyOption{&llb.CopyInfo{AllowWildcard: true}}
	outputFile := fmt.Sprintf("%s-%s.gguf", c.Output.Name, c.Output.Quantize)
	return llb.Scratch().File(llb.Copy(state, inputFile, outputFile, copyOpts...)), nil
}

func gpuCacheKey(opts Options) string {
	cdiDevice := opts.CDIDevice
	if cdiDevice == "" {
		cdiDevice = nvidiaCDIDevice
	}
	if opts.NVIDIADriverVersion != "" && isImmutableNVIDIACDIDevice(cdiDevice) {
		return fmt.Sprintf("device:%s;driver:%s", cdiDevice, opts.NVIDIADriverVersion)
	}
	if opts.BuildSessionID != "" {
		return "session:" + opts.BuildSessionID
	}
	return ""
}

func isImmutableNVIDIACDIDevice(cdiDevice string) bool {
	selector, found := strings.CutPrefix(cdiDevice, "nvidia.com/gpu=")
	if !found {
		return false
	}
	return strings.HasPrefix(selector, "GPU-") || strings.HasPrefix(selector, "MIG-")
}

func runUnslothPhase(state, scriptState, configState llb.State, phase, cdiDevice string, includeLlamaCache bool) llb.State {
	runOptions := []llb.RunOption{
		utils.Shf("ldconfig && nvidia-smi && %[1]s && python /aikit-bin/target_unsloth.py %[2]s", sourceVenv, phase),
		llb.AddCDIDevice(llb.CDIDeviceName(cdiDevice)),
		llb.AddMount("/aikit-bin", scriptState, llb.Readonly),
		llb.AddMount("/aikit-config", configState, llb.Readonly),
		persistentCacheMount("/root/.cache/huggingface", huggingFaceCacheID, llb.CacheMountShared),
		persistentCacheMount(datasetsCachePath, datasetsCacheID, llb.CacheMountShared),
		persistentCacheMount("/root/.cache/torch", torchCacheID, llb.CacheMountShared),
		persistentCacheMount("/root/.triton", tritonCacheID, llb.CacheMountShared),
	}
	if includeLlamaCache {
		runOptions = append(runOptions, persistentCacheMount("/root/.unsloth", llamaCacheID, llb.CacheMountShared))
	}
	return state.Run(runOptions...).Root()
}

func persistentCacheMount(target, id string, sharing llb.CacheMountSharingMode) llb.RunOption {
	return llb.AddMount(target, llb.Scratch(), llb.AsPersistentCacheDir(id, sharing))
}

func mustMarshalYAML(value interface{}) []byte {
	data, err := yaml.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
}
