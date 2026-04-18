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
	distrolessBase                = "ghcr.io/kaito-project/aikit/base:latest"
	localAIBinaryVersion          = "v4.0.0"
	localAILlamaCppBackendVersion = localAIBinaryVersion
	localAILegacyBackendVersion   = "v3.12.1"
	localAIROCmBackendVersion     = "rocm7"
	localAIRepo                   = "ghcr.io/kaito-project/aikit/localai:"
	cudaVersion                   = "12-5"
	rocmVersion                   = "7.2"
)

// Aikit2LLB converts an InferenceConfig to an LLB state.
func Aikit2LLB(c *config.InferenceConfig, platform *specs.Platform) (llb.State, *specs.Image, error) {
	var merge, state llb.State
	switch c.Runtime {
	case utils.RuntimeAppleSilicon:
		state = llb.Image(utils.AppleSiliconBase, llb.Platform(*platform))
	case utils.RuntimeROCm:
		// Use Ubuntu 24.04 for ROCm to match noble repository
		state = llb.Image(utils.Ubuntu24Base, llb.Platform(*platform))
	default:
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

	state, merge, err = addLocalAI(c, state, merge, *platform)
	if err != nil {
		return state, nil, err
	}

	// install cuda if runtime is nvidia and architecture is amd64
	if c.Runtime == utils.RuntimeNVIDIA && platform.Architecture == utils.PlatformAMD64 {
		state, merge = installCuda(c, state, merge)
	}

	// install rocm if runtime is rocm and architecture is amd64
	if c.Runtime == utils.RuntimeROCm && platform.Architecture == utils.PlatformAMD64 {
		state, merge = installRocm(c, state, merge)
	}

	// install backend dependencies
	merge = installBackends(c, *platform, state, merge)

	imageCfg := NewImageConfig(c, platform)
	return merge, imageCfg, nil
}

// getBaseImage returns the base image given the InferenceConfig and platform.
func getBaseImage(c *config.InferenceConfig, platform *specs.Platform) llb.State {
	if c.Runtime == utils.RuntimeAppleSilicon {
		return llb.Image(utils.AppleSiliconBase, llb.Platform(*platform))
	}
	if c.Runtime == utils.RuntimeROCm {
		// Use Ubuntu 24.04 for ROCm to match noble repository.
		return llb.Image(utils.Ubuntu24Base, llb.Platform(*platform))
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

func installRocm(c *config.InferenceConfig, s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s

	// Set up ROCm repository
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y ca-certificates curl gnupg"), llb.IgnoreCache).Root()

	// Add ROCm GPG key and repository
	s = s.Run(utils.Sh("curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg")).Root()
	s = s.Run(utils.Shf("echo 'deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm.gpg] https://repo.radeon.com/rocm/apt/%s/ noble main' >> /etc/apt/sources.list.d/rocm.list", rocmVersion)).Root()
	s = s.Run(utils.Shf("echo 'deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm.gpg] https://repo.radeon.com/graphics/%s/ubuntu noble main' >> /etc/apt/sources.list.d/rocm.list", rocmVersion)).Root()
	rocmPinning := `
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
`
	s = s.Run(utils.Shf("echo '%s' > /etc/apt/preferences.d/repo-radeon-pin-600", rocmPinning)).Root()
	s = s.Run(utils.Sh("apt-get update"), llb.IgnoreCache).Root()

	// default llama.cpp backend is being used
	if len(c.Backends) == 0 {
		// install rocm libraries and pciutils for gpu detection
		s = s.Run(utils.Sh("apt-get install -y pciutils rocm && apt-get clean")).Root()
	}

	// For backends that specify llama-cpp explicitly
	for b := range c.Backends {
		if c.Backends[b] == utils.BackendLlamaCpp {
			// Install ROCm libraries needed for llama-cpp ROCm acceleration
			rocmDeps := "apt-get install -y rocm && apt-get clean"
			s = s.Run(utils.Sh(rocmDeps)).Root()
		}
	}

	// hipblaslt soname compatibility: backend may be linked against .so.0 while ROCm 7.2 ships .so.1
	s = s.Run(utils.Sh("set -e; cd /opt/rocm/lib; [ -e libhipblaslt.so.0 ] || ln -sf libhipblaslt.so.1 libhipblaslt.so.0")).Root()

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

// addLocalAI adds the LocalAI binary to the image.
func addLocalAI(c *config.InferenceConfig, s llb.State, merge llb.State, platform specs.Platform) (llb.State, llb.State, error) {
	artifactVersion := getLocalAIArtifactVersion(c, platform)

	// Map architectures to OCI artifact references & internal artifact filenames
	artifactRefs := map[string]struct {
		Ref string
	}{
		utils.PlatformAMD64: {Ref: localAIRepo + artifactVersion + "-amd64"},
		utils.PlatformARM64: {Ref: localAIRepo + artifactVersion + "-arm64"},
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
