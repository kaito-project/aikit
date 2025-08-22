package build

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/containerd/platforms"
	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/aikit2llb/finetune"
	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/utils"
	controlapi "github.com/moby/buildkit/api/services/control"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	d2llb "github.com/moby/buildkit/frontend/dockerfile/dockerfile2llb"
	"github.com/moby/buildkit/frontend/dockerui"
	"github.com/moby/buildkit/frontend/gateway/client"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

const (
	localNameContext     = "context"
	localNameDockerfile  = "dockerfile"
	localNameAikitfile   = "aikitfile.yaml"
	defaultAikitfileName = "aikitfile.yaml"

	keyFilename       = "filename"
	keyTarget         = "target"
	keyOutput         = "output"
	keyTargetPlatform = "platform"
	keyCacheImports   = "cache-imports"
)

func Build(ctx context.Context, c client.Client) (*client.Result, error) {
	inferenceCfg, finetuneCfg, err := getAikitfileConfig(ctx, c)
	if err != nil {
		return nil, errors.Wrap(err, "getting aikitfile")
	}

	if finetuneCfg != nil {
		return buildFineTune(ctx, c, finetuneCfg)
	} else if inferenceCfg != nil {
		return buildInference(ctx, c, inferenceCfg)
	}

	return nil, nil
}

func buildFineTune(ctx context.Context, c client.Client, cfg *config.FineTuneConfig) (*client.Result, error) {
	err := validateFinetuneConfig(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "validating aikitfile")
	}

	// set defaults for unsloth and finetune config
	if cfg.Target == utils.TargetUnsloth {
		cfg = defaultsUnslothConfig(cfg)
	}
	cfg = defaultsFineTune(cfg)

	buildOpts := c.BuildOpts()
	opts := buildOpts.Opts

	// Parse cache imports
	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	st := finetune.Aikit2LLB(cfg)

	def, err := st.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal local source")
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
	err := validateInferenceConfig(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "validating aikitfile")
	}

	buildOpts := c.BuildOpts()
	opts := buildOpts.Opts

	// Parse cache imports
	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	// Default the build platform to the buildkit host's os/arch
	defaultBuildPlatform := platforms.DefaultSpec()

	// But prefer the first worker's platform
	if workers := c.BuildOpts().Workers; len(workers) > 0 && len(workers[0].Platforms) > 0 {
		defaultBuildPlatform = workers[0].Platforms[0]
	}

	buildPlatforms := []specs.Platform{defaultBuildPlatform}

	targetPlatforms := []*specs.Platform{nil}
	if platform, exists := opts[keyTargetPlatform]; exists && platform != "" {
		targetPlatforms, err = parsePlatforms(platform)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse target platforms %s", platform)
		}
	} else if platform == "" {
		targetPlatforms = []*specs.Platform{&defaultBuildPlatform}
	}

	// Validate backends against target platforms
	err = validateBackendPlatformCompatibility(cfg, targetPlatforms)
	if err != nil {
		return nil, errors.Wrap(err, "validating backend platform compatibility")
	}

	if cfg.Runtime == utils.RuntimeAppleSilicon {
		for _, tp := range targetPlatforms {
			if tp.Architecture != utils.PlatformARM64 {
				return nil, errors.New("apple silicon runtime only supports arm64 platform")
			}
		}
	}

	isMultiPlatform := len(targetPlatforms) > 1
	exportPlatforms := &exptypes.Platforms{
		Platforms: make([]exptypes.Platform, len(targetPlatforms)),
	}
	finalResult := client.NewResult()

	eg, ctx := errgroup.WithContext(ctx)

	// Solve for all target platforms in parallel
	for i, tp := range targetPlatforms {
		func(i int, platform *specs.Platform) {
			eg.Go(func() (err error) {
				result, err := buildImage(ctx, c, cfg, &d2llb.ConvertOpt{
					MetaResolver:   c,
					TargetPlatform: platform,
					Config: dockerui.Config{
						BuildPlatforms:         buildPlatforms,
						MultiPlatformRequested: isMultiPlatform,
						CacheImports:           cacheImports,
					},
				})
				if err != nil {
					return errors.Wrap(err, "failed to build image")
				}

				result.AddToClientResult(finalResult)
				exportPlatforms.Platforms[i] = result.ExportPlatform

				return nil
			})
		}(i, tp)
	}

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	if isMultiPlatform {
		dt, err := json.Marshal(exportPlatforms)
		if err != nil {
			return nil, err
		}
		finalResult.AddMeta(exptypes.ExporterPlatformsKey, dt)
	}

	return finalResult, nil
}

// Represents the result of a single image build.
type buildResult struct {
	// Reference to built image
	Reference client.Reference

	// Image configuration
	ImageConfig []byte

	// Target platform
	Platform *specs.Platform

	// Whether this is a result for a multi-platform build
	MultiPlatform bool

	// Exportable platform information (platform and platform ID)
	ExportPlatform exptypes.Platform
}

// AddToClientResult adds the build result to a client result.
func (br *buildResult) AddToClientResult(cr *client.Result) {
	if br.MultiPlatform {
		cr.AddMeta(
			fmt.Sprintf("%s/%s", exptypes.ExporterImageConfigKey, br.ExportPlatform.ID),
			br.ImageConfig,
		)
		cr.AddRef(br.ExportPlatform.ID, br.Reference)
	} else {
		cr.AddMeta(exptypes.ExporterImageConfigKey, br.ImageConfig)
		cr.SetRef(br.Reference)
	}
}

// buildImage builds an image from the given aikitfile config.
func buildImage(ctx context.Context, c client.Client, cfg *config.InferenceConfig, convertOpts *d2llb.ConvertOpt) (*buildResult, error) {
	result := buildResult{
		Platform:      convertOpts.TargetPlatform,
		MultiPlatform: convertOpts.MultiPlatformRequested,
	}

	state, image, err := inference.Aikit2LLB(cfg, convertOpts.TargetPlatform)
	if err != nil {
		return nil, err
	}

	result.ImageConfig, err = json.Marshal(image)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal image config")
	}

	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal definition")
	}

	res, err := c.Solve(ctx, client.SolveRequest{
		Definition:   def.ToPB(),
		CacheImports: convertOpts.CacheImports,
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to solve")
	}

	result.Reference, err = res.SingleRef()
	if err != nil {
		return nil, err
	}

	// Add platform-specific export info for the result that can later be used
	// in multi-platform results
	result.ExportPlatform = exptypes.Platform{
		Platform: platforms.DefaultSpec(),
	}

	if result.Platform != nil {
		result.ExportPlatform.Platform = *result.Platform
	}

	result.ExportPlatform.ID = platforms.Format(result.ExportPlatform.Platform)

	return &result, nil
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
	switch {
	case strings.HasPrefix(context, "git"):
		st, ok = dockerui.DetectGitContext(context, true)
		if !ok {
			return nil, nil, errors.Errorf("invalid git context %s", context)
		}
	case strings.HasPrefix(context, "http") || strings.HasPrefix(context, "https"):
		st, ok = dockerui.DetectGitContext(context, true)
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
		return nil, nil, errors.Wrapf(err, "failed to marshal local source")
	}

	res, err := c.Solve(ctx, client.SolveRequest{
		Definition: def.ToPB(),
	})
	if err != nil {
		return nil, nil, errors.Wrapf(err, "failed to resolve aikitfile")
	}

	ref, err := res.SingleRef()
	if err != nil {
		return nil, nil, err
	}

	dtAikitfile, err := ref.ReadFile(ctx, client.ReadRequest{
		Filename: filename,
	})
	if err != nil {
		return nil, nil, errors.Wrapf(err, "failed to read aikitfile")
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

	err = parseBuildArgs(opts, inferenceCfg)
	if err != nil {
		return nil, nil, errors.Wrap(err, "parsing build args")
	}

	return inferenceCfg, finetuneCfg, nil
}

// getBuildArg returns the value of the build arg with the given key.
func getBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}

// validateFinetuneConfig validates the finetune config.
func validateFinetuneConfig(c *config.FineTuneConfig) error {
	supportedFineTuneTargets := []string{utils.TargetUnsloth}

	if c.APIVersion == "" {
		return errors.New("apiVersion is not defined")
	}

	if c.APIVersion != utils.APIv1alpha1 {
		return errors.Errorf("apiVersion %s is not supported", c.APIVersion)
	}

	if !slices.Contains(supportedFineTuneTargets, c.Target) {
		return errors.Errorf("target %s is not supported", c.Target)
	}

	if len(c.Datasets) == 0 {
		return errors.New("no datasets defined")
	}

	if len(c.Datasets) > 1 {
		return errors.New("only one dataset is supported at this time")
	}

	// only alpaca dataset is supported at this time
	for _, d := range c.Datasets {
		if d.Type != utils.DatasetAlpaca {
			return errors.Errorf("dataset type %s is not supported", d.Type)
		}
	}
	return nil
}

// defaultsUnslothConfig sets default values for the unsloth config.
func defaultsUnslothConfig(c *config.FineTuneConfig) *config.FineTuneConfig {
	if c.Config.Unsloth.MaxSeqLength == 0 {
		c.Config.Unsloth.MaxSeqLength = 2048
	}
	if c.Config.Unsloth.BatchSize == 0 {
		c.Config.Unsloth.BatchSize = 2
	}
	if c.Config.Unsloth.GradientAccumulationSteps == 0 {
		c.Config.Unsloth.GradientAccumulationSteps = 4
	}
	if c.Config.Unsloth.WarmupSteps == 0 {
		c.Config.Unsloth.WarmupSteps = 10
	}
	if c.Config.Unsloth.MaxSteps == 0 {
		c.Config.Unsloth.MaxSteps = 60
	}
	if c.Config.Unsloth.LearningRate == 0 {
		c.Config.Unsloth.LearningRate = 0.0002
	}
	if c.Config.Unsloth.LoggingSteps == 0 {
		c.Config.Unsloth.LoggingSteps = 1
	}
	if c.Config.Unsloth.Optimizer == "" {
		c.Config.Unsloth.Optimizer = "adamw_8bit"
	}
	if c.Config.Unsloth.WeightDecay == 0 {
		c.Config.Unsloth.WeightDecay = 0.01
	}
	if c.Config.Unsloth.LrSchedulerType == "" {
		c.Config.Unsloth.LrSchedulerType = "linear"
	}
	if c.Config.Unsloth.Seed == 0 {
		c.Config.Unsloth.Seed = 42
	}
	return c
}

// defaultsFineTune sets default values for the fine-tune config.
func defaultsFineTune(c *config.FineTuneConfig) *config.FineTuneConfig {
	if c.Output.Quantize == "" {
		c.Output.Quantize = "q4_k_m"
	}
	if c.Output.Name == "" {
		c.Output.Name = "aikit-model"
	}
	return c
}

// validateInferenceConfig validates the inference config.
func validateInferenceConfig(c *config.InferenceConfig) error {
	if c.APIVersion == "" {
		return errors.New("apiVersion is not defined")
	}

	if c.APIVersion != utils.APIv1alpha1 {
		return errors.Errorf("apiVersion %s is not supported", c.APIVersion)
	}

	if len(c.Backends) > 1 {
		return errors.New("only one backend is supported at this time")
	}

	if slices.Contains(c.Backends, utils.BackendDiffusers) && c.Runtime != utils.RuntimeNVIDIA {
		return errors.New("diffusers backend only supports nvidia cuda runtime. please add 'runtime: cuda' to your aikitfile.yaml")
	}

	if c.Runtime == utils.RuntimeAppleSilicon && len(c.Backends) > 0 {
		for _, backend := range c.Backends {
			if backend != utils.BackendLlamaCpp {
				return errors.New("apple silicon runtime only supports llama-cpp backend")
			}
		}
	}

	backends := []string{utils.BackendLlamaCpp, utils.BackendExllamaV2, utils.BackendDiffusers}
	for _, b := range c.Backends {
		if !slices.Contains(backends, b) {
			return errors.Errorf("backend %s is not supported", b)
		}
	}

	runtimes := []string{"", utils.RuntimeNVIDIA, utils.RuntimeAppleSilicon}
	if !slices.Contains(runtimes, c.Runtime) {
		return errors.Errorf("runtime %s is not supported", c.Runtime)
	}

	return nil
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

// parsePlatforms parses a comma-separated list of platforms.
func parsePlatforms(v string) ([]*specs.Platform, error) {
	var pp []*specs.Platform
	for _, v := range strings.Split(v, ",") {
		p, err := platforms.Parse(v)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse target platform %s", v)
		}
		p = platforms.Normalize(p)
		pp = append(pp, &p)
	}
	return pp, nil
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
