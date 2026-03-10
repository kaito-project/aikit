package inference

import (
	"fmt"
	"net/url"
	"slices"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	distrolessBase = "ghcr.io/kaito-project/aikit/base:latest"
	localAIVersion = "v3.12.1"
	localAIRepo    = "ghcr.io/kaito-project/aikit/localai:"
	cudaVersion    = "12-5"
)

// Aikit2LLB converts an InferenceConfig to an LLB state.
func Aikit2LLB(c *config.InferenceConfig, platform *specs.Platform) (llb.State, *specs.Image, error) {
	var merge, state llb.State
	if c.Runtime == utils.RuntimeAppleSilicon {
		state = llb.Image(utils.AppleSiliconBase, llb.Platform(*platform))
	} else {
		state = llb.Image(utils.UbuntuBase, llb.Platform(*platform))
	}
	base := getBaseImage(c, platform)

	var err error
	if isRunnerMode(c) {
		// Runner mode: skip model downloads, write config if present, install runner deps
		state, merge = writeConfig(c, base, state, *platform)
		state, merge = installRunnerDependencies(c, state, merge, *platform)
		state, merge = installRunnerEntrypoint(c, state, merge)
	} else {
		// Standard mode: download models + write config
		state, merge, err = copyModels(c, base, state, *platform)
		if err != nil {
			return state, nil, err
		}
	}

	state, merge, err = addLocalAI(state, merge, *platform)
	if err != nil {
		return state, nil, err
	}

	// install cuda if runtime is nvidia and architecture is amd64
	if c.Runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		state, merge = installCuda(c, state, merge)
	}

	// install backend dependencies
	merge = installBackends(c, *platform, state, merge)

	// install GPU detection wrapper for CUDA images to work around
	// LocalAI v3.12.1 bug where /usr/local/cuda-12 directory presence
	// causes CUDA backend selection even without a GPU.
	// See: https://github.com/mudler/LocalAI/pull/6149
	if c.Runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		state, merge = installGPUDetectionWrapper(state, merge)
	}

	imageCfg := NewImageConfig(c, platform)
	return merge, imageCfg, nil
}

// getBaseImage returns the base image given the InferenceConfig and platform.
func getBaseImage(c *config.InferenceConfig, platform *specs.Platform) llb.State {
	if c.Runtime == utils.RuntimeAppleSilicon {
		return llb.Image(utils.AppleSiliconBase, llb.Platform(*platform))
	}
	if len(c.Backends) > 0 {
		return llb.Image(utils.UbuntuBase, llb.Platform(*platform))
	}
	return llb.Image(distrolessBase, llb.Platform(*platform))
}

// writeConfig writes the /config.yaml file to the image when c.Config is set.
func writeConfig(c *config.InferenceConfig, base llb.State, s llb.State, platform specs.Platform) (llb.State, llb.State) {
	savedState := s
	if c.Config != "" {
		s = s.File(
			llb.Mkfile("/config.yaml", 0o644, []byte(c.Config)),
			llb.WithCustomName(fmt.Sprintf("Creating config for platform %s/%s", platform.OS, platform.Architecture)),
		)
	}
	diff := llb.Diff(savedState, s)
	merge := llb.Merge([]llb.State{base, diff})
	return s, merge
}

// copyModels copies models to the image and writes the config.
func copyModels(c *config.InferenceConfig, base llb.State, s llb.State, platform specs.Platform) (llb.State, llb.State, error) {
	savedState := s
	for _, model := range c.Models {
		// Check if the model source is a URL
		if _, err := url.ParseRequestURI(model.Source); err == nil {
			switch {
			case strings.HasPrefix(model.Source, "oci://"):
				s = handleOCI(model.Source, s, platform)
			case strings.HasPrefix(model.Source, "http://"), strings.HasPrefix(model.Source, "https://"):
				s = handleHTTP(model.Source, model.Name, model.SHA256, s)
			case strings.HasPrefix(model.Source, "huggingface://"):
				s, err = handleHuggingFace(model.Source, s)
				if err != nil {
					return llb.State{}, llb.State{}, err
				}
			default:
				return llb.State{}, llb.State{}, fmt.Errorf("unsupported URL scheme: %s", model.Source)
			}
		} else {
			// Handle local paths
			s = handleLocal(model.Source, s)
		}

		// create prompt templates if defined
		for _, pt := range model.PromptTemplates {
			if pt.Name != "" && pt.Template != "" {
				s = s.Run(utils.Shf("echo -n \"%s\" > /models/%s.tmpl", pt.Template, pt.Name)).Root()
			}
		}
	}

	// create config file if defined
	if c.Config != "" {
		s = s.Run(utils.Shf("mkdir -p /configuration && echo -n \"%s\" > /config.yaml", c.Config),
			llb.WithCustomName(fmt.Sprintf("Creating config for platform %s/%s", platform.OS, platform.Architecture))).Root()
	}

	diff := llb.Diff(savedState, s)
	merge := llb.Merge([]llb.State{base, diff})
	return s, merge, nil
}

// installCuda installs cuda libraries and dependencies.
func installCuda(c *config.InferenceConfig, s llb.State, merge llb.State) (llb.State, llb.State) {
	cudaKeyringURL := "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
	cudaKeyring := llb.HTTP(cudaKeyringURL)
	s = s.File(
		llb.Copy(cudaKeyring, utils.FileNameFromURL(cudaKeyringURL), "/"),
		llb.WithCustomName("Copying "+utils.FileNameFromURL(cudaKeyringURL)), //nolint: goconst
	)
	s = s.Run(utils.Sh("dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb")).Root()

	savedState := s
	// running apt-get update twice due to nvidia repo
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y ca-certificates && apt-get update"), llb.IgnoreCache).Root()

	// install cuda libraries for llama-cpp (default) and vllm backends
	if len(c.Backends) == 0 || slices.Contains(c.Backends, utils.BackendLlamaCpp) || slices.Contains(c.Backends, utils.BackendVLLM) {
		// install cuda libraries and pciutils for gpu detection
		s = s.Run(utils.Shf("apt-get install -y --no-install-recommends pciutils libcublas-%[1]s cuda-cudart-%[1]s && apt-get clean", cudaVersion)).Root()
		// TODO: clean up /var/lib/dpkg/status
	}

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

// gpuDetectionWrapper is a shell script that detects GPU presence at container
// startup. If no NVIDIA GPU is found, it forces LocalAI to use the CPU backend
// by setting LOCALAI_FORCE_META_BACKEND_CAPABILITY=default. This works around
// a LocalAI v3.12.1 regression where the existence of /usr/local/cuda-12
// (installed by CUDA runtime packages) causes LocalAI to select the CUDA
// backend even when no GPU hardware is present.
const gpuDetectionWrapper = `#!/bin/sh
# Detect NVIDIA GPU and set backend capability accordingly.
# If no GPU is found, force LocalAI to use CPU backends.
if command -v lspci >/dev/null 2>&1; then
  if ! lspci -d 10de: 2>/dev/null | grep -qi 'nvidia\|3d controller\|vga'; then
    export LOCALAI_FORCE_META_BACKEND_CAPABILITY=default
  fi
elif ! [ -e /dev/nvidiactl ]; then
  export LOCALAI_FORCE_META_BACKEND_CAPABILITY=default
fi
exec "$@"
`

// installGPUDetectionWrapper writes the GPU detection entrypoint wrapper into the image.
func installGPUDetectionWrapper(s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s
	s = s.File(
		llb.Mkfile("/usr/local/bin/gpu-detect-wrapper", 0o755, []byte(gpuDetectionWrapper)),
		llb.WithCustomName("Installing GPU detection wrapper for CPU fallback"),
	)
	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

// addLocalAI adds the LocalAI binary to the image.
func addLocalAI(s llb.State, merge llb.State, platform specs.Platform) (llb.State, llb.State, error) {
	// Map architectures to OCI artifact references & internal artifact filenames
	artifactRefs := map[string]struct {
		Ref string
	}{
		utils.PlatformAMD64: {Ref: localAIRepo + localAIVersion + "-amd64"},
		utils.PlatformARM64: {Ref: localAIRepo + localAIVersion + "-arm64"},
	}

	art, ok := artifactRefs[platform.Architecture]
	if !ok {
		return s, merge, fmt.Errorf("unsupported architecture %s", platform.Architecture)
	}

	savedState := s

	// Use the oras CLI image to pull the artifact containing the LocalAI binary
	tooling := llb.Image(orasImage, llb.Platform(platform)).Run(
		utils.Shf("set -e\noras pull %[1]s\nchmod +x local-ai\nchmod 755 local-ai", art.Ref),
		llb.WithCustomName("Pulling LocalAI from OCI artifact "+art.Ref),
	).Root()

	// Copy the prepared binary into /usr/bin/local-ai
	s = s.File(
		llb.Copy(tooling, "local-ai", "/usr/bin/local-ai"),
		llb.WithCustomName("Copying local-ai from OCI artifact to /usr/bin"),
	)

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff}), nil
}
