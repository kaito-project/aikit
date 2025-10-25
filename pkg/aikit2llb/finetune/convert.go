package finetune

import (
	"fmt"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/kaito-project/aikit/pkg/version"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/util/system"
	"gopkg.in/yaml.v2"
)

const (
	unslothCommitOrTag = "fb77505f8429566f5d21d6ea5318c342e8a67991" // September-2024
	nvidiaMknod        = "mknod --mode 666 /dev/nvidiactl c 195 255 && mknod --mode 666 /dev/nvidia-uvm c 235 0 && mknod --mode 666 /dev/nvidia-uvm-tools c 235 1 && mknod --mode 666 /dev/nvidia0 c 195 0 && nvidia-smi"
	sourceVenv         = ". .venv/bin/activate"
)

func Aikit2LLB(c *config.FineTuneConfig) llb.State {
	env := map[string]string{
		"PATH":                       system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin",
		"NVIDIA_REQUIRE_CUDA":        "cuda>=12.0",
		"NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
		"NVIDIA_VISIBLE_DEVICES":     "all",
		"LD_LIBRARY_PATH":            "/usr/local/cuda/lib64",
	}

	state := llb.Image(utils.CudaDevel)
	for k, v := range env {
		state = state.AddEnv(k, v)
	}

	// installing dependencies
	// due to buildkit run limitations, we need to install nvidia drivers and driver version must match the host
	state = state.Run(utils.Sh("apt-get update && apt-get install -y --no-install-recommends python3-dev python3 python3-pip python-is-python3 git wget kmod && cd /root && VERSION=$(cat /proc/driver/nvidia/version | sed -n 's/.*NVIDIA UNIX x86_64 Kernel Module  \\([0-9]\\+\\.[0-9]\\+\\.[0-9]\\+\\).*/\\1/p') && wget --no-verbose https://download.nvidia.com/XFree86/Linux-x86_64/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run && chmod +x NVIDIA-Linux-x86_64-$VERSION.run && ./NVIDIA-Linux-x86_64-$VERSION.run -x && rm NVIDIA-Linux-x86_64-$VERSION.run && /root/NVIDIA-Linux-x86_64-$VERSION/nvidia-installer -a -s --skip-depmod --no-dkms --no-nvidia-modprobe --no-questions --no-systemd --no-x-check --no-kernel-modules --no-kernel-module-source && rm -rf /root/NVIDIA-Linux-x86_64-$VERSION")).Root()

	// write config to /config.yaml
	cfg, err := yaml.Marshal(c)
	if err != nil {
		panic(err)
	}
	state = state.Run(utils.Shf("echo -n \"%s\" > /config.yaml", string(cfg))).Root()

	var scratch llb.State
	if c.Target == utils.TargetUnsloth {
		// installing unsloth and its dependencies
		// uv does not support installing xformers via unsloth pyproject
		state = state.Run(utils.Shf("pip install --upgrade pip uv && uv venv --system-site-packages && %[1]s && uv pip install --upgrade --force-reinstall packaging torch==2.4.0 ipython ninja packaging bitsandbytes setuptools==69.5.1 wheel psutil transformers==4.44.2 numpy==2.0.2 && uv pip install flash-attn --no-build-isolation && python -m pip install 'unsloth[cu121_ampere_torch240] @ git+https://github.com/unslothai/unsloth.git@%[2]s'", sourceVenv, unslothCommitOrTag)).Root()

		version := version.Version
		if version == "" {
			version = "main"
		}
		unslothScriptURL := fmt.Sprintf("https://raw.githubusercontent.com/kaito-project/aikit/%s/pkg/finetune/target_unsloth.py", version)
		var opts []llb.HTTPOption
		opts = append(opts, llb.Chmod(0o755))
		unslothScript := llb.HTTP(unslothScriptURL, opts...)
		state = state.File(
			llb.Copy(unslothScript, utils.FileNameFromURL(unslothScriptURL), "/"),
			llb.WithCustomName("Copying "+utils.FileNameFromURL(unslothScriptURL)),
		)

		// setup nvidia devices and run unsloth
		// due to buildkit run limitations, we need to create the devices manually and run unsloth in the same command
		state = state.Run(utils.Shf("%[1]s && %[2]s && python -m target_unsloth", nvidiaMknod, sourceVenv), llb.Security(llb.SecurityModeInsecure)).Root()

		// copy gguf to scratch which will be the output
		const inputFile = "model/*.gguf"
		copyOpts := []llb.CopyOption{}
		copyOpts = append(copyOpts, &llb.CopyInfo{AllowWildcard: true})
		outputFile := fmt.Sprintf("%s-%s.gguf", c.Output.Name, c.Output.Quantize)
		scratch = llb.Scratch().File(llb.Copy(state, inputFile, outputFile, copyOpts...))
	}

	return scratch
}
