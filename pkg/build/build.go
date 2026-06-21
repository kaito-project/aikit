package build

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/aikit2llb/finetune"
	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/packager"
	"github.com/kaito-project/aikit/pkg/utils"
	controlapi "github.com/moby/buildkit/api/services/control"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/dockerui"
	"github.com/moby/buildkit/frontend/gateway/client"
	dockerspec "github.com/moby/docker-image-spec/specs-go/v1"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

const (
	localNameContext     = "context"
	localNameDockerfile  = "dockerfile"
	defaultAikitfileName = "aikitfile.yaml"

	keyFilename       = "filename"
	keyTarget         = "target"
	keyOutput         = "output"
	keyTargetPlatform = "platform"
	keyCacheImports   = "cache-imports"

	packagerTargetPrefix = "packager/"
)

// Build is the BuildKit frontend entrypoint. It recovers from panics in build
// logic so that a programming error surfaces as a build failure with a stack
// trace rather than crashing the frontend process.
func Build(ctx context.Context, c client.Client) (_ *client.Result, retErr error) {
	defer func() {
		if r := recover(); r != nil {
			retErr = errors.Errorf("recovered from panic in frontend: %+v\n%s", r, getPanicStack())
		}
	}()

	opts := c.BuildOpts().Opts

	// Packager targets are dispatched directly. An unknown packager/* target is
	// an explicit error rather than a silent fall-through to aikitfile loading.
	if t := opts[keyTarget]; strings.HasPrefix(t, packagerTargetPrefix) {
		switch t {
		case "packager/modelpack":
			return packager.BuildModelpack(ctx, c)
		case "packager/generic":
			return packager.BuildGeneric(ctx, c)
		default:
			return nil, errors.Errorf("unknown target %q", t)
		}
	}

	inferenceCfg, finetuneCfg, err := getAikitfileConfig(ctx, c)
	if err != nil {
		return nil, errors.Wrap(err, "getting aikitfile")
	}

	switch {
	case finetuneCfg != nil:
		return buildFineTune(ctx, c, finetuneCfg)
	case inferenceCfg != nil:
		return buildInference(ctx, c, inferenceCfg)
	default:
		return nil, errors.New("aikitfile did not contain a valid inference or finetune configuration")
	}
}

// getPanicStack captures the current goroutine's stack as an error-friendly string.
func getPanicStack() string {
	stackBuf := make([]uintptr, 32)
	n := runtime.Callers(3, stackBuf) // Skip runtime.Callers, getPanicStack, and the deferred closure.
	stackBuf = stackBuf[:n]
	frames := runtime.CallersFrames(stackBuf)
	var sb strings.Builder
	for {
		frame, more := frames.Next()
		fmt.Fprintf(&sb, "%s\n\t%s:%d\n", frame.Function, frame.File, frame.Line)
		if !more {
			break
		}
	}
	return sb.String()
}

func buildFineTune(ctx context.Context, c client.Client, cfg *config.FineTuneConfig) (*client.Result, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(err, "validating aikitfile")
	}

	cfg.FillDefaults()

	opts := c.BuildOpts().Opts

	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	st, err := finetune.Aikit2LLB(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "failed to build finetune llb")
	}

	def, err := st.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal local source")
	}
	res, err := c.Solve(ctx, client.SolveRequest{
		Definition:   def.ToPB(),
		CacheImports: cacheImports,
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to solve")
	}
	return res, nil
}

func buildInference(ctx context.Context, c client.Client, cfg *config.InferenceConfig) (*client.Result, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(err, "validating aikitfile")
	}

	opts := c.BuildOpts().Opts

	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	// dockerui.NewClient resolves the build platform (preferring the first
	// worker's platform) and parses requested target platforms from opts,
	// replacing the previously hand-rolled platform handling.
	dc, err := dockerui.NewClient(c)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create dockerui client")
	}

	// When no target platform is requested, build for the build platform so the
	// exported image targets the worker/host platform (prior behavior).
	if len(dc.TargetPlatforms) == 0 {
		dc.TargetPlatforms = dc.BuildPlatforms
	}

	targetPlatforms := make([]*specs.Platform, len(dc.TargetPlatforms))
	for i := range dc.TargetPlatforms {
		p := dc.TargetPlatforms[i]
		targetPlatforms[i] = &p
	}

	if err := validateBackendPlatformCompatibility(cfg, targetPlatforms); err != nil {
		return nil, errors.Wrap(err, "validating backend platform compatibility")
	}

	if cfg.Runtime == utils.RuntimeAppleSilicon {
		for _, tp := range targetPlatforms {
			if tp.Architecture != utils.PlatformARM64 {
				return nil, errors.New("apple silicon runtime only supports arm64 platform")
			}
		}
	}

	rb, err := dc.Build(ctx, func(ctx context.Context, platform *specs.Platform, _ int) (*dockerui.BuildResult, error) {
		return buildImage(ctx, c, cfg, platform, cacheImports)
	})
	if err != nil {
		return nil, err
	}

	return rb.Finalize()
}

// buildImage builds an image from the given aikitfile config for one platform.
func buildImage(ctx context.Context, c client.Client, cfg *config.InferenceConfig, platform *specs.Platform, cacheImports []client.CacheOptionsEntry) (*dockerui.BuildResult, error) {
	state, image, err := inference.Aikit2LLB(cfg, platform)
	if err != nil {
		return nil, err
	}

	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal definition")
	}

	res, err := c.Solve(ctx, client.SolveRequest{
		Definition:   def.ToPB(),
		CacheImports: cacheImports,
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to solve")
	}

	ref, err := res.SingleRef()
	if err != nil {
		return nil, err
	}

	return &dockerui.BuildResult{
		Reference: ref,
		Image:     toDockerImage(image),
	}, nil
}

// toDockerImage converts an OCI image spec into the Docker-extended image spec
// expected by dockerui. The Docker-specific extension fields are all omitempty
// and left unset by aikit, so the marshaled config is byte-identical to the OCI
// image config. The outer Config field must be set explicitly because it shadows
// the embedded ocispec.Image.Config at JSON-marshal time.
func toDockerImage(img *specs.Image) *dockerspec.DockerOCIImage {
	if img == nil {
		return nil
	}
	return &dockerspec.DockerOCIImage{
		Image: *img,
		Config: dockerspec.DockerOCIImageConfig{
			ImageConfig: img.Config,
		},
	}
}

func getAikitfileConfig(ctx context.Context, c client.Client) (*config.InferenceConfig, *config.FineTuneConfig, error) {
	opts := c.BuildOpts().Opts
	filename := opts[keyFilename]
	if filename == "" {
		filename = defaultAikitfileName
	}

	name := "load aikitfile"
	if filename != "aikitfile.yaml" {
		name += " from " + filename
	}

	context := opts[localNameContext]

	var st *llb.State
	var ok bool
	keepGit := true
	switch {
	case strings.HasPrefix(context, "git"):
		st, ok, _ = dockerui.DetectGitContext(context, &keepGit)
		if !ok {
			return nil, nil, errors.Errorf("invalid git context %s", context)
		}
	case strings.HasPrefix(context, "http") || strings.HasPrefix(context, "https"):
		st, ok, _ = dockerui.DetectGitContext(context, &keepGit)
		if !ok {
			st, filename, _ = dockerui.DetectHTTPContext(context)
		}
	default:
		localSt := llb.Local(localNameDockerfile,
			llb.IncludePatterns([]string{filename}),
			llb.SessionID(c.BuildOpts().SessionID),
			llb.SharedKeyHint(defaultAikitfileName),
			dockerui.WithInternalName(name),
		)
		st = &localSt
	}

	def, err := st.Marshal(ctx)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to marshal local source")
	}

	res, err := c.Solve(ctx, client.SolveRequest{
		Definition: def.ToPB(),
	})
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to resolve aikitfile")
	}

	ref, err := res.SingleRef()
	if err != nil {
		return nil, nil, err
	}

	dtAikitfile, err := ref.ReadFile(ctx, client.ReadRequest{
		Filename: filename,
	})
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read aikitfile")
	}

	inferenceCfg, finetuneCfg, err := config.NewFromBytes(dtAikitfile)
	if err != nil {
		return nil, nil, errors.Wrap(err, "getting config")
	}
	if finetuneCfg != nil {
		target, ok := opts[keyTarget]
		if !ok {
			target = utils.TargetUnsloth
		}
		finetuneCfg.Target = target

		if opts[keyOutput] != "" {
			return nil, nil, errors.New("--output is required for finetune. please specify a directory to save the finetuned model")
		}
	}

	if err := parseBuildArgs(opts, inferenceCfg); err != nil {
		return nil, nil, errors.Wrap(err, "parsing build args")
	}

	return inferenceCfg, finetuneCfg, nil
}

// validateBackendPlatformCompatibility validates that backends are compatible with target platforms.
func validateBackendPlatformCompatibility(c *config.InferenceConfig, targetPlatforms []*specs.Platform) error {
	// Check if any target platform is ARM64
	hasARM64Platform := false
	for _, tp := range targetPlatforms {
		if tp != nil && tp.Architecture == utils.PlatformARM64 {
			hasARM64Platform = true
			break
		}
	}

	// ROCm runtime only supports amd64.
	if c.Runtime == utils.RuntimeROCm && hasARM64Platform {
		return errors.New("rocm runtime is only supported on linux/amd64 platform")
	}

	// If we have ARM64 platforms, validate backend compatibility
	if hasARM64Platform {
		for _, backend := range c.Backends {
			if backend != utils.BackendLlamaCpp {
				return errors.Errorf("backend %s is not supported on arm64 platform. only llama-cpp backend supports arm64", backend)
			}
		}
	}

	return nil
}

// parseCacheOptions handles given cache imports.
func parseCacheOptions(opts map[string]string) ([]client.CacheOptionsEntry, error) {
	var cacheImports []client.CacheOptionsEntry
	if cacheImportsStr := opts[keyCacheImports]; cacheImportsStr != "" {
		var cacheImportsUM []*controlapi.CacheOptionsEntry
		if err := json.Unmarshal([]byte(cacheImportsStr), &cacheImportsUM); err != nil {
			return nil, errors.Wrapf(err, "failed to unmarshal %s (%q)", keyCacheImports, cacheImportsStr)
		}
		for _, um := range cacheImportsUM {
			cacheImports = append(cacheImports, client.CacheOptionsEntry{Type: um.Type, Attrs: um.Attrs})
		}
	}
	return cacheImports, nil
}
