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
func Aikit2LLB(c *config.InferenceConfig, targetPlatform *specs.Platform) (llb.State, *specs.Image, error) {
	return Aikit2LLBWithPlatforms(c, targetPlatform, targetPlatform)
}

// Aikit2LLBWithPlatforms converts an InferenceConfig using separate build and target platforms.
func Aikit2LLBWithPlatforms(c *config.InferenceConfig, buildPlatform, targetPlatform *specs.Platform) (llb.State, *specs.Image, error) {
	if buildPlatform == nil {
		buildPlatform = targetPlatform
	}

	var merge, state llb.State
	switch c.Runtime {
	case utils.RuntimeAppleSilicon:
		state = llb.Image(utils.AppleSiliconBase, llb.Platform(*targetPlatform))
	case utils.RuntimeROCm:
		// Use Ubuntu 24.04 for ROCm to match noble repository.
		state = llb.Image(utils.Ubuntu24Base, llb.Platform(*targetPlatform))
	default:
		state = llb.Image(utils.UbuntuBase, llb.Platform(*targetPlatform))
	}
	buildBase := state
	base := getBaseImage(c, targetPlatform)

	var err error
	if isRunnerMode(c) {
		// Runner mode skips model downloads and keeps dependencies and the entrypoint sequential.
		_, merge = writeConfig(c, base, buildBase, *targetPlatform)
		state, merge = installRunnerDependencies(c, buildBase, merge, *targetPlatform)
		state, merge = installRunnerEntrypoint(c, state, merge)
	} else {
		// Standard mode materializes models and config on an isolated branch.
		state, merge, err = copyModels(c, base, buildBase, *buildPlatform, *targetPlatform)
		if err != nil {
			return state, nil, err
		}
		state = buildBase
	}

	state, merge, err = addLocalAI(c, state, merge, *buildPlatform, *targetPlatform)
	if err != nil {
		return state, nil, err
	}

	// install cuda if runtime is nvidia and architecture is amd64
	if c.Runtime == utils.RuntimeNVIDIA && targetPlatform.Architecture == utils.PlatformAMD64 {
		state, merge = installCuda(c, state, merge)
	}

	// install rocm if runtime is rocm and architecture is amd64
	if c.Runtime == utils.RuntimeROCm && targetPlatform.Architecture == utils.PlatformAMD64 {
		state, merge = installRocm(c, state, merge)
	}

	// install backend dependencies
	merge = installBackends(c, *targetPlatform, state, merge)

	imageCfg := NewImageConfig(c, targetPlatform)
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
	return applyAndMerge(s, base, func(s llb.State) llb.State {
		if c.Config != "" {
			s = s.File(
				llb.Mkfile("/config.yaml", 0o644, []byte(c.Config)),
				llb.WithCustomName(fmt.Sprintf("Creating config for platform %s/%s", platform.OS, platform.Architecture)),
			)
		}
		return s
	})
}

// copyModels copies models to the image and writes the config.
func copyModels(c *config.InferenceConfig, base llb.State, s llb.State, buildPlatform, targetPlatform specs.Platform) (llb.State, llb.State, error) {
	savedState := s
	localSources := make([]string, 0, len(c.Models))
	for _, model := range c.Models {
		parsedSource, err := url.ParseRequestURI(model.Source)
		if err != nil || parsedSource.Scheme == "" {
			localSources = append(localSources, model.Source)
		}
	}
	localContext := localModelContext(localSources)

	var configurationFiles *llb.FileAction
	for _, model := range c.Models {
		parsedSource, err := url.ParseRequestURI(model.Source)
		if err != nil || parsedSource.Scheme == "" {
			// Parse failures and empty schemes are local paths, including absolute paths.
			s = handleLocal(model.Source, localContext, s)
		} else {
			parsedSource.Scheme = strings.ToLower(parsedSource.Scheme)
			normalizedSource := parsedSource.String()
			switch parsedSource.Scheme {
			case "oci":
				s = handleOCI(normalizedSource, s, buildPlatform, targetPlatform)
			case "http", "https":
				s = handleHTTP(normalizedSource, model.Name, model.SHA256, s)
			case "huggingface":
				s, err = handleHuggingFace(normalizedSource, s)
				if err != nil {
					return llb.State{}, llb.State{}, err
				}
			default:
				return llb.State{}, llb.State{}, fmt.Errorf("unsupported URL scheme: %s", model.Source)
			}
		}

		// create prompt templates if defined
		for _, pt := range model.PromptTemplates {
			if pt.Name != "" && pt.Template != "" {
				path := fmt.Sprintf("/models/%s.tmpl", pt.Name)
				if configurationFiles == nil {
					configurationFiles = llb.Mkfile(path, 0o644, []byte(pt.Template))
				} else {
					configurationFiles = configurationFiles.Mkfile(path, 0o644, []byte(pt.Template))
				}
			}
		}
	}

	// create config file if defined
	if c.Config != "" {
		if configurationFiles == nil {
			configurationFiles = llb.Mkdir("/configuration", 0o755, llb.WithParents(true))
		} else {
			configurationFiles = configurationFiles.Mkdir("/configuration", 0o755, llb.WithParents(true))
		}
		configurationFiles = configurationFiles.Mkfile("/config.yaml", 0o644, []byte(c.Config))
	}

	if configurationFiles != nil {
		var opts []llb.ConstraintsOpt
		if c.Config != "" {
			opts = append(opts, llb.WithCustomName(fmt.Sprintf("Creating config for platform %s/%s", targetPlatform.OS, targetPlatform.Architecture)))
		}
		s = s.File(configurationFiles, opts...)
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

	return applyAndMerge(s, merge, func(s llb.State) llb.State {
		// running apt-get update twice due to nvidia repo
		s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y ca-certificates && apt-get update"), llb.IgnoreCache).Root()

		// install cuda libraries for llama-cpp (default) and vllm backends
		if len(c.Backends) == 0 || slices.Contains(c.Backends, utils.BackendLlamaCpp) || slices.Contains(c.Backends, utils.BackendVLLM) {
			// install cuda libraries and pciutils for gpu detection
			s = s.Run(utils.Shf("apt-get install -y --no-install-recommends pciutils libcublas-%[1]s cuda-cudart-%[1]s && apt-get clean", cudaVersion)).Root()
			// TODO: clean up /var/lib/dpkg/status
		}
		return s
	})
}

func installRocm(c *config.InferenceConfig, s llb.State, merge llb.State) (llb.State, llb.State) {
	return applyAndMerge(s, merge, func(s llb.State) llb.State {
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

		// install rocm libraries and pciutils for gpu detection when using the default
		// llama-cpp backend or when it is configured explicitly
		if len(c.Backends) == 0 || slices.Contains(c.Backends, utils.BackendLlamaCpp) {
			s = s.Run(utils.Sh("apt-get install -y pciutils rocm && apt-get clean")).Root()
		}

		// hipblaslt soname compatibility: backend may be linked against .so.0 while ROCm 7.2 ships .so.1
		s = s.Run(utils.Sh("set -e; cd /opt/rocm/lib; [ -e libhipblaslt.so.0 ] || ln -sf libhipblaslt.so.1 libhipblaslt.so.0")).Root()
		return s
	})
}

// addLocalAI adds the LocalAI binary to the image.
func addLocalAI(c *config.InferenceConfig, s llb.State, merge llb.State, buildPlatform, targetPlatform specs.Platform) (llb.State, llb.State, error) {
	artifactVersion := getLocalAIArtifactVersion(c, targetPlatform)

	// Map architectures to OCI artifact references & internal artifact filenames
	artifactRefs := map[string]struct {
		Ref string
	}{
		utils.PlatformAMD64: {Ref: localAIRepo + artifactVersion + "-amd64"},
		utils.PlatformARM64: {Ref: localAIRepo + artifactVersion + "-arm64"},
	}

	art, ok := artifactRefs[targetPlatform.Architecture]
	if !ok {
		return s, merge, fmt.Errorf("unsupported architecture %s", targetPlatform.Architecture)
	}

	// Use the oras CLI image to pull the artifact containing the LocalAI binary
	tooling := llb.Image(orasImage, llb.Platform(buildPlatform)).Run(
		utils.Shf("set -e\noras pull %[1]s\nchmod +x local-ai\nchmod 755 local-ai", art.Ref),
		llb.WithCustomName("Pulling LocalAI from OCI artifact "+art.Ref),
	).Root()

	// Copy the prepared binary into /usr/bin/local-ai
	s, merge = applyAndMerge(s, merge, func(s llb.State) llb.State {
		return s.File(
			llb.Copy(tooling, "local-ai", "/usr/bin/local-ai"),
			llb.WithCustomName("Copying local-ai from OCI artifact to /usr/bin"),
		)
	})
	return s, merge, nil
}
