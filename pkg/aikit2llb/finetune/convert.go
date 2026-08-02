package finetune

import (
	"fmt"

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
	sourceVenv     = ". .venv/bin/activate"

	nvidiaMknod = "NVIDIA_MAJOR=$(awk '$2 == \"nvidia\" { print $1; exit }' /proc/devices) && " +
		"NVIDIA_UVM_MAJOR=$(awk '$2 == \"nvidia-uvm\" { print $1; exit }' /proc/devices) && " +
		"test -n \"$NVIDIA_MAJOR\" && test -n \"$NVIDIA_UVM_MAJOR\" && " +
		"rm -f /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 && " +
		"mknod --mode 666 /dev/nvidiactl c \"$NVIDIA_MAJOR\" 255 && " +
		"mknod --mode 666 /dev/nvidia-uvm c \"$NVIDIA_UVM_MAJOR\" 0 && " +
		"mknod --mode 666 /dev/nvidia-uvm-tools c \"$NVIDIA_UVM_MAJOR\" 1 && " +
		"mknod --mode 666 /dev/nvidia0 c \"$NVIDIA_MAJOR\" 0 && nvidia-smi"
)

func Aikit2LLB(c *config.FineTuneConfig) llb.State {
	env := []struct {
		key   string
		value string
	}{
		{key: "PATH", value: system.DefaultPathEnv("linux") + ":/usr/local/cuda/bin"},
		{key: "NVIDIA_REQUIRE_CUDA", value: "cuda>=12.6"},
		{key: "NVIDIA_DRIVER_CAPABILITIES", value: "compute,utility"},
		{key: "NVIDIA_VISIBLE_DEVICES", value: "all"},
		{key: "LD_LIBRARY_PATH", value: "/usr/local/cuda/lib64"},
	}

	state := llb.Image(utils.CudaDevel)
	for _, entry := range env {
		state = state.AddEnv(entry.key, entry.value)
	}

	// installing dependencies
	// due to buildkit run limitations, we need to install nvidia drivers and driver version must match the host
	state = state.Run(utils.Sh("apt-get update && apt-get install -y --no-install-recommends cmake python3-dev python3 python3-pip python-is-python3 git wget kmod && cd /root && VERSION=$(cat /proc/driver/nvidia/version | sed -n 's/.*NVIDIA UNIX x86_64 Kernel Module  \\([0-9]\\+\\.[0-9]\\+\\.[0-9]\\+\\).*/\\1/p') && wget --no-verbose https://download.nvidia.com/XFree86/Linux-x86_64/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run && chmod +x NVIDIA-Linux-x86_64-$VERSION.run && ./NVIDIA-Linux-x86_64-$VERSION.run -x && rm NVIDIA-Linux-x86_64-$VERSION.run && /root/NVIDIA-Linux-x86_64-$VERSION/nvidia-installer -a -s --skip-depmod --no-dkms --no-nvidia-modprobe --no-questions --no-systemd --no-x-check --no-kernel-modules --no-kernel-module-source && rm -rf /root/NVIDIA-Linux-x86_64-$VERSION")).Root()

	var scratch llb.State
	if c.Target == utils.TargetUnsloth {
		// Install the latest tested Unsloth release and its matching CUDA 12.6/PyTorch 2.10 dependencies.
		state = state.Run(utils.Shf(
			"pip install --upgrade pip uv && "+
				"uv venv --system-site-packages && %[1]s && "+
				"uv pip install --torch-backend=cu126 'torch==%[2]s' "+
				"'unsloth[cu126-torch2100]==%[3]s' 'unsloth-zoo==%[3]s'",
			sourceVenv,
			torchVersion,
			unslothVersion,
		)).Root()

		state = state.File(
			llb.Mkfile("/target_unsloth.py", 0o755, finetunescript.TargetUnsloth),
			llb.WithCustomName("Copying target_unsloth.py"),
		)
	}

	// Write config after invariant setup so config-only changes reuse dependency layers.
	cfg, err := yaml.Marshal(c)
	if err != nil {
		panic(err)
	}
	state = state.File(llb.Mkfile("/config.yaml", 0o644, cfg))

	if c.Target == utils.TargetUnsloth {
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
