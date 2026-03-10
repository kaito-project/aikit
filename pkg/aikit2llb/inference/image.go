package inference

import (
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func NewImageConfig(c *config.InferenceConfig, platform *specs.Platform) *specs.Image {
	img := emptyImage(c, platform)

	// For CUDA images on amd64, prepend the GPU detection wrapper to the
	// entrypoint. This wrapper detects whether an NVIDIA GPU is actually
	// present at runtime and forces LocalAI to use CPU backends if not,
	// working around a LocalAI v3.12.1 regression where /usr/local/cuda-12
	// directory presence alone causes CUDA backend selection.
	// The wrapper is only installed for amd64 (see installGPUDetectionWrapper
	// gating in convert.go), so we must match that condition here.
	gpuWrapper := c.Runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64

	if isRunnerMode(c) {
		// Runner mode: use the aikit-runner entrypoint script
		if gpuWrapper {
			img.Config.Entrypoint = []string{"/usr/local/bin/gpu-detect-wrapper", "/usr/local/bin/aikit-runner"}
		} else {
			img.Config.Entrypoint = []string{"/usr/local/bin/aikit-runner"}
		}
		img.Config.Cmd = []string{}

		// Add runner labels
		backendLabel := strings.Join(c.Backends, ",")
		img.Config.Labels = map[string]string{
			"ai.kaito.aikit.runner":  "true",
			"ai.kaito.aikit.backend": backendLabel,
		}
		if c.Runtime != "" {
			img.Config.Labels["ai.kaito.aikit.runtime"] = c.Runtime
		}
	} else {
		// Standard mode: use local-ai directly
		cmd := []string{}
		if c.Debug {
			cmd = append(cmd, "--debug")
		}
		if c.Config != "" {
			cmd = append(cmd, "--config-file=/config.yaml")
		}

		if gpuWrapper {
			img.Config.Entrypoint = []string{"/usr/local/bin/gpu-detect-wrapper", "local-ai"}
		} else {
			img.Config.Entrypoint = []string{"local-ai"}
		}
		img.Config.Cmd = cmd
	}

	return img
}

func emptyImage(c *config.InferenceConfig, platform *specs.Platform) *specs.Image {
	img := &specs.Image{
		Platform: specs.Platform{
			Architecture: platform.Architecture,
			OS:           utils.PlatformLinux,
		},
	}
	img.RootFS.Type = "layers"
	img.Config.WorkingDir = "/"

	img.Config.Env = []string{
		"PATH=" + system.DefaultPathEnv(utils.PlatformLinux),
		"CONFIG_FILE=/config.yaml",
	}

	cudaEnv := []string{
		"PATH=" + system.DefaultPathEnv(utils.PlatformLinux) + ":/usr/local/cuda/bin",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_VISIBLE_DEVICES=all",
		"LD_LIBRARY_PATH=/usr/local/cuda/lib64",
		"BUILD_TYPE=cublas",
		"CUDA_HOME=/usr/local/cuda",
	}
	if c.Runtime == utils.RuntimeNVIDIA {
		img.Config.Env = append(img.Config.Env, cudaEnv...)
	}

	if c.Runtime == utils.RuntimeAppleSilicon {
		img.Config.Env = append(img.Config.Env,
			"VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json",
		)
	}

	return img
}
