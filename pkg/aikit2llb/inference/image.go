package inference

import (
	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func NewImageConfig(c *config.InferenceConfig, platform *specs.Platform) *specs.Image {
	img := emptyImage(c, platform)
	cmd := []string{}
	if c.Debug {
		cmd = append(cmd, "--debug")
	}
	if c.Config != "" {
		cmd = append(cmd, "--config-file=/config.yaml")
	}

	img.Config.Entrypoint = []string{"local-ai"}
	img.Config.Cmd = cmd
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
		"CUDA_MAJOR_VERSION=12",
	}
	if c.Runtime == utils.RuntimeNVIDIA {
		img.Config.Env = append(img.Config.Env, cudaEnv...)
	}

	return img
}
