package inference

import (
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	localAIEntrypointCommand = "local-ai"
	localAILoadToMemoryEnv   = "LOCALAI_LOAD_TO_MEMORY="
	runnerEntrypointPath     = "/usr/local/bin/aikit-runner"
	runnerHFHomeEnv          = "HF_HOME=/models/.cache/huggingface"
)

func NewImageConfig(c *config.InferenceConfig, platform *specs.Platform) *specs.Image {
	backend, err := ResolveBackend(c, *platform)
	if err != nil {
		panic("resolving backend for image config: " + err.Error())
	}

	return NewImageConfigWithBackend(c, backend, platform)
}

// NewImageConfigWithBackend creates image metadata from a pre-resolved backend plan.
func NewImageConfigWithBackend(c *config.InferenceConfig, backend backendcatalog.Resolution, platform *specs.Platform) *specs.Image {
	img := emptyImage(c, backend, platform)
	runtimeBase := runtimeBaseForConfig(c, backend)
	img.Config.Labels = map[string]string{
		"ai.kaito.aikit.backend":                backend.Family,
		"ai.kaito.aikit.backend.artifact":       backend.Backend.Ref,
		"ai.kaito.aikit.backend.catalog.digest": backend.CatalogDigest,
		"ai.kaito.aikit.backend.selector":       string(backend.Selector),
		"ai.kaito.aikit.backend.status":         string(backend.Status),
		"ai.kaito.aikit.core.artifact":          backend.Core.Ref,
		"ai.kaito.aikit.runtime-base.artifact":  runtimeBase.Ref,
	}

	if isRunnerMode(c) {
		// Runner mode: use the aikit-runner entrypoint script
		img.Config.Entrypoint = []string{runnerEntrypointPath}
		img.Config.Cmd = []string{}
		img.Config.Env = append(img.Config.Env, runnerHFHomeEnv)

		// Add runner labels
		img.Config.Labels["ai.kaito.aikit.runner"] = "true"
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

		img.Config.Entrypoint = []string{localAIEntrypointCommand}
		img.Config.Cmd = cmd
	}

	return img
}

func emptyImage(c *config.InferenceConfig, backend backendcatalog.Resolution, platform *specs.Platform) *specs.Image {
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
	if len(c.LoadToMemory) > 0 {
		img.Config.Env = append(img.Config.Env, localAILoadToMemoryEnv+strings.Join(c.LoadToMemory, ","))
	}
	img.Config.Env = append(img.Config.Env, backend.Environment...)

	return img
}
