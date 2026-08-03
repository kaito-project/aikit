package finetune

import (
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	finetunescript "github.com/kaito-project/aikit/pkg/finetune"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/util/system"
	"gopkg.in/yaml.v2"
)

const (
	unslothVersion = "2026.8.1"
	torchVersion   = "2.10.0"
	uvVersion      = "0.12.1"
	pythonVenv     = "/opt/aikit-venv"
	sourceVenv     = ". " + pythonVenv + "/bin/activate"

	aptCacheID          = "aikit-finetune-apt-v1"
	aptListsCacheID     = "aikit-finetune-apt-lists-v1"
	nvidiaDriverCacheID = "aikit-finetune-nvidia-driver-v1"
	huggingFaceCacheID  = "aikit-unsloth-huggingface-v1"
	datasetsCacheID     = "aikit-unsloth-datasets-v1"
	torchCacheID        = "aikit-unsloth-torch-v1"
	tritonCacheID       = "aikit-unsloth-triton-v1"
	llamaCacheID        = "aikit-unsloth-llama-v1"
	uvCacheID           = "aikit-unsloth-uv-" + uvVersion + "-py310-cu126-torch-" + torchVersion + "-unsloth-" + unslothVersion
	datasetsCachePath   = "/tmp/aikit-datasets-cache"

	nvidiaPrimaryMajorAWK = `$2 == "nvidia-frontend" { frontend = $1 } $2 == "nvidia" { legacy = $1 } ` +
		`END { if (frontend != "") { print frontend } else if (legacy != "") { print legacy } ` +
		`else { print "failed to find NVIDIA primary character device major (expected nvidia-frontend or nvidia)" > "/dev/stderr"; exit 1 } }`
	nvidiaMknod = "NVIDIA_MAJOR=$(awk '" + nvidiaPrimaryMajorAWK + "' /proc/devices) && " +
		"NVIDIA_UVM_MAJOR=$(awk '$2 == \"nvidia-uvm\" { print $1; exit }' /proc/devices) && " +
		"if [ -z \"$NVIDIA_UVM_MAJOR\" ]; then echo \"failed to find NVIDIA UVM character device major\" >&2; exit 1; fi && " +
		"rm -f /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 && " +
		"mknod --mode 666 /dev/nvidiactl c \"$NVIDIA_MAJOR\" 255 && " +
		"mknod --mode 666 /dev/nvidia-uvm c \"$NVIDIA_UVM_MAJOR\" 0 && " +
		"mknod --mode 666 /dev/nvidia-uvm-tools c \"$NVIDIA_UVM_MAJOR\" 1 && " +
		"mknod --mode 666 /dev/nvidia0 c \"$NVIDIA_MAJOR\" 0 && nvidia-smi"
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

// Options configures host-specific fine-tuning build behavior.
type Options struct {
	NVIDIADriverVersion string
}

func Aikit2LLB(c *config.FineTuneConfig, options ...Options) llb.State {
	var opts Options
	if len(options) > 0 {
		opts = options[0]
	}

	env := []struct {
		key   string
		value string
	}{
		{key: "PATH", value: system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin"},
		{key: "NVIDIA_REQUIRE_CUDA", value: "cuda>=12.6"},
		{key: "NVIDIA_DRIVER_CAPABILITIES", value: "compute,utility"},
		{key: "NVIDIA_VISIBLE_DEVICES", value: "all"},
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

	// Keep OS package installation cacheable and independent of the host driver version.
	state = state.Run(
		utils.Sh("rm -f /etc/apt/apt.conf.d/docker-clean && apt-get update && apt-get install -y --no-install-recommends cmake git kmod mawk python-is-python3 python3 python3-dev python3-pip wget"),
		persistentCacheMount("/var/cache/apt", aptCacheID, llb.CacheMountLocked),
		persistentCacheMount("/var/lib/apt/lists", aptListsCacheID, llb.CacheMountLocked),
	).Root()

	state = state.Run(nvidiaDriverInstallOptions(opts.NVIDIADriverVersion)...).Root()

	if c.Target != utils.TargetUnsloth {
		return llb.Scratch()
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

	scriptState := llb.Scratch().File(llb.Mkfile("/target_unsloth.py", 0o755, finetunescript.TargetUnsloth))
	trainingConfig := unslothTrainingConfig{
		BaseModel: c.BaseModel,
		Datasets:  c.Datasets,
		Config:    c.Config,
	}
	trainingConfigState := llb.Scratch().File(llb.Mkfile("/train-config.yaml", 0o600, mustMarshalYAML(trainingConfig)))
	state = runUnslothPhase(state, scriptState, trainingConfigState, "train", false)

	exportConfig := unslothExportConfig{
		BaseModel: c.BaseModel,
		Config:    c.Config,
	}
	exportConfig.Output.Quantize = c.Output.Quantize
	exportConfigState := llb.Scratch().File(llb.Mkfile("/export-config.yaml", 0o600, mustMarshalYAML(exportConfig)))
	state = runUnslothPhase(state, scriptState, exportConfigState, "export", true)

	const inputFile = "model/*.gguf"
	copyOpts := []llb.CopyOption{&llb.CopyInfo{AllowWildcard: true}}
	outputFile := fmt.Sprintf("%s-%s.gguf", c.Output.Name, c.Output.Quantize)
	return llb.Scratch().File(llb.Copy(state, inputFile, outputFile, copyOpts...))
}

func nvidiaDriverInstallOptions(version string) []llb.RunOption {
	resolveVersion := `VERSION=$(sed -n 's/.*NVIDIA UNIX x86_64 Kernel Module  \([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)`
	if version != "" {
		resolveVersion = "VERSION=" + shellQuote(version)
	}

	command := fmt.Sprintf(`%s &&
if [ -z "$VERSION" ]; then
	echo "failed to resolve NVIDIA driver version from /proc/driver/nvidia/version; expose the host NVIDIA procfs entry or pass --build-arg nvidiaDriverVersion=<major.minor.patch>" >&2
	exit 1
fi &&
DRIVER_CACHE=/root/.cache/nvidia-driver &&
INSTALLER_NAME=NVIDIA-Linux-x86_64-$VERSION.run &&
INSTALLER=$DRIVER_CACHE/$INSTALLER_NAME &&
DOWNLOAD_URL=https://download.nvidia.com/XFree86/Linux-x86_64/$VERSION/$INSTALLER_NAME &&
CHECKSUM_URL=$DOWNLOAD_URL.sha256sum &&
mkdir -p "$DRIVER_CACHE" &&
DOWNLOAD_DIR=$(mktemp -d "$DRIVER_CACHE/.${INSTALLER_NAME}.download.XXXXXX") &&
INSTALLER_TMP=$DOWNLOAD_DIR/$INSTALLER_NAME &&
CHECKSUM_TMP=$DOWNLOAD_DIR/$INSTALLER_NAME.sha256sum &&
trap 'rm -rf "$DOWNLOAD_DIR"' 0 HUP INT TERM &&
wget --no-verbose "$CHECKSUM_URL" -O "$CHECKSUM_TMP" &&
if [ -s "$INSTALLER" ] && (cd "$DRIVER_CACHE" && sha256sum --check --status "$CHECKSUM_TMP"); then
	:
else
	rm -f "$INSTALLER" &&
	wget --no-verbose "$DOWNLOAD_URL" -O "$INSTALLER_TMP" &&
	(cd "$DOWNLOAD_DIR" && sha256sum --check --status "$CHECKSUM_TMP") &&
	chmod +x "$INSTALLER_TMP" &&
	mv "$INSTALLER_TMP" "$INSTALLER"
fi &&
rm -rf "$DOWNLOAD_DIR" &&
trap - 0 HUP INT TERM &&
chmod +x "$INSTALLER" &&
cd /root &&
rm -rf "NVIDIA-Linux-x86_64-$VERSION" &&
"$INSTALLER" -x &&
/root/NVIDIA-Linux-x86_64-$VERSION/nvidia-installer -a -s --skip-depmod --no-dkms --no-nvidia-modprobe --no-questions --no-systemd --no-x-check --no-kernel-modules --no-kernel-module-source &&
rm -rf /root/NVIDIA-Linux-x86_64-$VERSION`, resolveVersion)
	options := []llb.RunOption{
		utils.Sh(command),
		persistentCacheMount("/root/.cache/nvidia-driver", nvidiaDriverCacheID, llb.CacheMountLocked),
	}
	if version == "" {
		options = append(options, llb.IgnoreCache)
	}
	return options
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func runUnslothPhase(state, scriptState, configState llb.State, phase string, includeLlamaCache bool) llb.State {
	runOptions := []llb.RunOption{
		utils.Shf("%[1]s && %[2]s && python /aikit-bin/target_unsloth.py %[3]s", nvidiaMknod, sourceVenv, phase),
		llb.AddMount("/aikit-bin", scriptState, llb.Readonly),
		llb.AddMount("/aikit-config", configState, llb.Readonly),
		persistentCacheMount("/root/.cache/huggingface", huggingFaceCacheID, llb.CacheMountShared),
		persistentCacheMount(datasetsCachePath, datasetsCacheID, llb.CacheMountShared),
		persistentCacheMount("/root/.cache/torch", torchCacheID, llb.CacheMountShared),
		persistentCacheMount("/root/.triton", tritonCacheID, llb.CacheMountShared),
		llb.Security(llb.SecurityModeInsecure),
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
