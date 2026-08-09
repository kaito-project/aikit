package inference

import (
	"fmt"
	"net/url"
	"reflect"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const standardRuntimeCABundlePath = "/etc/ssl/certs/ca-certificates.crt"

// Aikit2LLB converts an InferenceConfig to an LLB state.
func Aikit2LLB(c *config.InferenceConfig, targetPlatform *specs.Platform) (llb.State, *specs.Image, error) {
	return Aikit2LLBWithPlatforms(c, targetPlatform, targetPlatform)
}

// Aikit2LLBWithPlatforms converts an InferenceConfig using separate build and target platforms.
func Aikit2LLBWithPlatforms(c *config.InferenceConfig, buildPlatform, targetPlatform *specs.Platform) (llb.State, *specs.Image, error) {
	if targetPlatform == nil {
		return llb.State{}, nil, fmt.Errorf("target platform is required")
	}

	backend, err := ResolveBackend(c, *targetPlatform)
	if err != nil {
		return llb.State{}, nil, err
	}

	return aikit2LLBWithResolvedBackend(c, buildPlatform, targetPlatform, backend)
}

// Aikit2LLBWithBackend converts an InferenceConfig using a pre-resolved immutable backend plan.
func Aikit2LLBWithBackend(c *config.InferenceConfig, buildPlatform, targetPlatform *specs.Platform, backend backendcatalog.Resolution) (llb.State, *specs.Image, error) {
	if targetPlatform == nil {
		return llb.State{}, nil, fmt.Errorf("target platform is required")
	}
	expected, err := ResolveBackend(c, *targetPlatform)
	if err != nil {
		return llb.State{}, nil, err
	}
	if !reflect.DeepEqual(backend, expected) {
		return llb.State{}, nil, fmt.Errorf("pre-resolved backend plan does not match the embedded catalog")
	}

	return aikit2LLBWithResolvedBackend(c, buildPlatform, targetPlatform, backend)
}

func aikit2LLBWithResolvedBackend(c *config.InferenceConfig, buildPlatform, targetPlatform *specs.Platform, backend backendcatalog.Resolution) (llb.State, *specs.Image, error) {
	if buildPlatform == nil {
		buildPlatform = targetPlatform
	}

	var merge llb.State
	runtimeBase := runtimeBaseForConfig(c, backend)
	state := llb.Image(runtimeBase.Ref, llb.Platform(*targetPlatform))
	buildBase := state
	base := state

	var err error
	if isRunnerMode(c) {
		// Runner mode skips model downloads and keeps dependencies and the entrypoint sequential.
		_, merge = writeConfig(c, base, buildBase, *targetPlatform)
		state, merge = installRunnerDependenciesWithBackend(backend, buildBase, merge, *targetPlatform)
		state, merge = installRunnerEntrypoint(c, backend, state, merge)
	} else {
		// Standard mode materializes models and config on an isolated branch.
		state, merge, err = copyModels(c, base, buildBase, *buildPlatform, *targetPlatform)
		if err != nil {
			return state, nil, err
		}
		state = buildBase
		state, merge = installStandardRuntimeTrust(state, merge, *buildPlatform)
	}

	state, merge, err = addLocalAI(backend, state, merge, *buildPlatform)
	if err != nil {
		return state, nil, err
	}

	// Install the exact backend artifacts selected during catalog preflight.
	merge = installBackends(backend, *targetPlatform, state, merge)
	if len(c.Models) > 0 {
		merge = installBackendModelAliases(backend, localAIModelDirectories(c.Config, c.Models), *buildPlatform, merge)
	}

	imageCfg := NewImageConfigWithBackend(c, backend, targetPlatform)
	return merge, imageCfg, nil
}

// installStandardRuntimeTrust copies the CA bundle from the existing build helper
// into standard images without assuming that the catalog runtime base has a package manager.
func installStandardRuntimeTrust(s llb.State, merge llb.State, buildPlatform specs.Platform) (llb.State, llb.State) {
	savedState := s
	trustSource := llb.Image(orasImage, llb.Platform(buildPlatform))
	s = s.File(
		llb.Copy(
			trustSource,
			standardRuntimeCABundlePath,
			standardRuntimeCABundlePath,
			&llb.CopyInfo{CreateDestPath: true},
		),
		llb.WithCustomName("Installing standard runtime CA roots"),
	)

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

func runtimeBaseForConfig(c *config.InferenceConfig, backend backendcatalog.Resolution) backendcatalog.Artifact {
	if isRunnerMode(c) && backend.RunnerRuntimeBase != nil {
		return *backend.RunnerRuntimeBase
	}

	return backend.RuntimeBase
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
func copyModels(c *config.InferenceConfig, base llb.State, s llb.State, buildPlatform, targetPlatform specs.Platform) (llb.State, llb.State, error) {
	savedState := s
	localSources := make([]string, 0, len(c.Models))
	for _, model := range c.Models {
		if _, err := url.ParseRequestURI(model.Source); err != nil {
			localSources = append(localSources, model.Source)
		}
	}
	localContext := localModelContext(localSources)

	var configurationFiles *llb.FileAction
	for _, model := range c.Models {
		// Check if the model source is a URL
		if _, err := url.ParseRequestURI(model.Source); err == nil {
			switch {
			case strings.HasPrefix(model.Source, "oci://"):
				s = handleOCI(model.Source, s, buildPlatform, targetPlatform)
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
			// Handle local paths.
			s = handleLocal(model.Source, localContext, s)
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

// addLocalAI adds the LocalAI binary to the image.
func addLocalAI(backend backendcatalog.Resolution, s llb.State, merge llb.State, buildPlatform specs.Platform) (llb.State, llb.State, error) {
	savedState := s

	// Use the oras CLI image to pull the artifact containing the LocalAI binary
	tooling := llb.Image(orasImage, llb.Platform(buildPlatform)).Run(
		utils.Shf("set -e\noras pull %[1]s\nchmod +x local-ai\nchmod 755 local-ai", backend.Core.Ref),
		llb.WithCustomName("Pulling LocalAI from OCI artifact "+backend.Core.Ref),
	).Root()

	// Copy the prepared binary into /usr/bin/local-ai
	s = s.File(
		llb.Copy(tooling, "local-ai", "/usr/bin/local-ai"),
		llb.WithCustomName("Copying local-ai from OCI artifact to /usr/bin"),
	)

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff}), nil
}
