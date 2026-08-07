package build

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"github.com/containerd/platforms"
	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/aikit2llb/finetune"
	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/packager"
	"github.com/kaito-project/aikit/pkg/utils"
	controlapi "github.com/moby/buildkit/api/services/control"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	d2llb "github.com/moby/buildkit/frontend/dockerfile/dockerfile2llb"
	"github.com/moby/buildkit/frontend/dockerui"
	"github.com/moby/buildkit/frontend/gateway/client"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
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
	nvidiaUUIDPattern = `[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}`
)

var (
	nvidiaDriverVersionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+(?:\.[0-9]+)?$`)
	nvidiaCDIDevicePattern     = regexp.MustCompile(`^nvidia\.com/gpu(?:=(?:all|[0-9]+(?::[0-9]+)?|gpu[0-9]+|mig[0-9]+:[0-9]+|(?:GPU|MIG)-` + nvidiaUUIDPattern + `))?$`)
	datasetSplitPattern        = regexp.MustCompile(`^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$`)
	datasetRevisionPattern     = regexp.MustCompile(`^[0-9a-f]{40}$`)
	datasetChecksumPattern     = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

func Build(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	if t, ok := opts[keyTarget]; ok {
		switch t {
		case "packager/modelpack":
			return packager.BuildModelpack(ctx, c)
		case "packager/generic":
			return packager.BuildGeneric(ctx, c)
		}
	}

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
	for _, warning := range datasetReproducibilityWarnings(cfg) {
		logrus.Warn(warning)
	}

	buildOpts := c.BuildOpts()
	opts := buildOpts.Opts
	finetuneOpts, err := parseFineTuneBuildOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "parsing fine-tune build options")
	}
	finetuneOpts.BuildSessionID = buildOpts.SessionID

	// Parse cache imports
	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	st, err := finetune.Aikit2LLB(cfg, finetuneOpts)
	if err != nil {
		return nil, errors.Wrap(err, "converting fine-tune config to LLB")
	}

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

	buildPlatform := buildPlatformFromConvertOpt(convertOpts)
	state, image, err := inference.Aikit2LLBWithPlatforms(cfg, &buildPlatform, convertOpts.TargetPlatform)
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

func buildPlatformFromConvertOpt(convertOpts *d2llb.ConvertOpt) specs.Platform {
	if len(convertOpts.BuildPlatforms) > 0 {
		return convertOpts.BuildPlatforms[0]
	}
	if convertOpts.TargetPlatform != nil {
		return *convertOpts.TargetPlatform
	}
	return platforms.DefaultSpec()
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

func parseFineTuneBuildOptions(opts map[string]string) (finetune.Options, error) {
	driverVersion := strings.TrimSpace(getBuildArg(opts, "nvidiaDriverVersion"))
	if driverVersion != "" && !nvidiaDriverVersionPattern.MatchString(driverVersion) {
		return finetune.Options{}, errors.Errorf("nvidiaDriverVersion %q must use major.minor or major.minor.patch format", driverVersion)
	}
	cdiDevice := strings.TrimSpace(getBuildArg(opts, "cdiDevice"))
	if cdiDevice != "" && !nvidiaCDIDevicePattern.MatchString(cdiDevice) {
		return finetune.Options{}, errors.Errorf("cdiDevice %q must be an NVIDIA GPU CDI device name", cdiDevice)
	}
	return finetune.Options{NVIDIADriverVersion: driverVersion, CDIDevice: cdiDevice}, nil
}

// validateFinetuneConfig validates the finetune config.
func validateFinetuneConfig(c *config.FineTuneConfig) error {
	if c == nil {
		return errors.New("fine-tune config is not defined")
	}

	if c.APIVersion == "" {
		return errors.New("apiVersion is not defined")
	}

	if c.APIVersion != utils.APIv1alpha1 {
		return errors.Errorf("apiVersion %s is not supported", c.APIVersion)
	}

	if c.Target != utils.TargetUnsloth {
		return errors.Errorf("target %s is not supported", c.Target)
	}

	if strings.TrimSpace(c.BaseModel) == "" {
		return errors.New("baseModel is not defined")
	}

	if len(c.Datasets) == 0 {
		return errors.New("no datasets defined")
	}

	for datasetIndex, dataset := range c.Datasets {
		if strings.TrimSpace(dataset.Source) == "" {
			return errors.Errorf("datasets[%d].source is not defined", datasetIndex)
		}
		switch dataset.Type {
		case utils.DatasetAlpaca, utils.DatasetMessages, utils.DatasetPromptCompletion, utils.DatasetShareGPT, utils.DatasetText:
		default:
			return errors.Errorf("datasets[%d].type %s is not supported", datasetIndex, dataset.Type)
		}
		if err := validateDatasetLoader(datasetIndex, dataset); err != nil {
			return err
		}
	}

	unsloth := c.Config.Unsloth
	if strings.TrimSpace(unsloth.Loss) == "" {
		return errors.New("config.unsloth.loss is not defined")
	}
	if unsloth.Loss != utils.SFTLossAll && unsloth.Loss != utils.SFTLossResponse {
		return errors.Errorf("config.unsloth.loss %s is not supported", unsloth.Loss)
	}
	if err := validateSFTDatasetCompatibility(c.Datasets, unsloth.Loss); err != nil {
		return err
	}
	if unsloth.Loss == utils.SFTLossResponse && unsloth.Packing {
		return errors.New("config.unsloth.loss response does not support packing because response masks must not cross conversation boundaries")
	}
	if unsloth.MaxSeqLength <= 0 {
		return errors.New("config.unsloth.maxSeqLength must be greater than zero")
	}
	if unsloth.BatchSize <= 0 {
		return errors.New("config.unsloth.batchSize must be greater than zero")
	}
	if unsloth.GradientAccumulationSteps <= 0 {
		return errors.New("config.unsloth.gradientAccumulationSteps must be greater than zero")
	}
	if unsloth.WarmupSteps < 0 {
		return errors.New("config.unsloth.warmupSteps must be zero or greater")
	}
	if unsloth.MaxSteps <= 0 {
		return errors.New("config.unsloth.maxSteps must be greater than zero")
	}
	if unsloth.LearningRate <= 0 || math.IsNaN(unsloth.LearningRate) || math.IsInf(unsloth.LearningRate, 0) {
		return errors.New("config.unsloth.learningRate must be a finite value greater than zero")
	}
	if unsloth.LoggingSteps <= 0 {
		return errors.New("config.unsloth.loggingSteps must be greater than zero")
	}
	if strings.TrimSpace(unsloth.Optimizer) == "" {
		return errors.New("config.unsloth.optimizer is not defined")
	}
	if !isSupportedUnslothOptimizer(unsloth.Optimizer) {
		return errors.Errorf("config.unsloth.optimizer %s is not supported", unsloth.Optimizer)
	}
	if unsloth.WeightDecay < 0 || math.IsNaN(unsloth.WeightDecay) || math.IsInf(unsloth.WeightDecay, 0) {
		return errors.New("config.unsloth.weightDecay must be a finite value zero or greater")
	}
	if strings.TrimSpace(unsloth.LrSchedulerType) == "" {
		return errors.New("config.unsloth.lrSchedulerType is not defined")
	}
	if !isSupportedUnslothScheduler(unsloth.LrSchedulerType) {
		return errors.Errorf("config.unsloth.lrSchedulerType %s is not supported", unsloth.LrSchedulerType)
	}
	if unsloth.Seed < 0 {
		return errors.New("config.unsloth.seed must be zero or greater")
	}

	if strings.TrimSpace(c.Output.Quantize) == "" {
		return errors.New("output.quantize is not defined")
	}
	normalizedQuantization := strings.ToLower(c.Output.Quantize)
	if !isSupportedUnslothQuantization(normalizedQuantization) {
		return errors.Errorf("output.quantize %q is not supported", c.Output.Quantize)
	}
	c.Output.Quantize = normalizedQuantization
	if !isPathSafeOutputName(c.Output.Name) {
		return errors.New("output name must be a safe filename containing only letters, numbers, dots, hyphens, or underscores")
	}

	return nil
}

type sftDatasetCompatibility string

const (
	sftCompatibilityFullSequence     sftDatasetCompatibility = "full-sequence"
	sftCompatibilityPromptCompletion sftDatasetCompatibility = "completion-only"
	sftCompatibilityResponseChat     sftDatasetCompatibility = "response-only chat"
)

func validateSFTDatasetCompatibility(datasets []config.Dataset, loss string) error {
	firstCompatibility, err := sftDatasetCompatibilityFor(datasets[0].Type, loss)
	if err != nil {
		return errors.Errorf("datasets[0] type %s: %s", datasets[0].Type, err)
	}

	for datasetIndex := 1; datasetIndex < len(datasets); datasetIndex++ {
		dataset := datasets[datasetIndex]
		compatibility, compatibilityErr := sftDatasetCompatibilityFor(dataset.Type, loss)
		if compatibilityErr != nil {
			return errors.Errorf("datasets[%d] type %s: %s", datasetIndex, dataset.Type, compatibilityErr)
		}
		if compatibility != firstCompatibility {
			return errors.Errorf(
				"datasets[%d] type %s is incompatible with datasets[0] type %s: %s and %s datasets cannot be combined",
				datasetIndex,
				dataset.Type,
				datasets[0].Type,
				compatibility,
				firstCompatibility,
			)
		}
	}

	return nil
}

func sftDatasetCompatibilityFor(datasetType, loss string) (sftDatasetCompatibility, error) {
	if loss == utils.SFTLossResponse {
		if datasetType != utils.DatasetMessages && datasetType != utils.DatasetShareGPT {
			return "", errors.New("config.unsloth.loss response is supported only for messages and sharegpt datasets")
		}
		return sftCompatibilityResponseChat, nil
	}

	if datasetType == utils.DatasetPromptCompletion {
		return sftCompatibilityPromptCompletion, nil
	}
	return sftCompatibilityFullSequence, nil
}

func validateDatasetLoader(datasetIndex int, dataset config.Dataset) error {
	loader := dataset.Loader
	if loader == nil {
		return nil
	}

	path := fmt.Sprintf("datasets[%d].loader", datasetIndex)
	if strings.TrimSpace(loader.Type) == "" {
		return errors.Errorf("%s.type is not defined", path)
	}
	if strings.TrimSpace(loader.Split) == "" {
		return errors.Errorf("%s.split is not defined", path)
	}
	if !datasetSplitPattern.MatchString(loader.Split) {
		return errors.Errorf("%s.split must be a named split containing letters, numbers, or underscores in dot-separated segments", path)
	}
	if loader.Subset != "" && strings.TrimSpace(loader.Subset) == "" {
		return errors.Errorf("%s.subset must not be empty", path)
	}

	switch loader.Type {
	case utils.DatasetLoaderHuggingFace:
		if isHTTPDatasetSource(dataset.Source) {
			return errors.Errorf("%s type huggingface does not support an HTTP(S) source", path)
		}
		if loader.Checksum != "" {
			return errors.Errorf("%s.checksum is not supported for type huggingface", path)
		}
		if loader.Revision != "" && !datasetRevisionPattern.MatchString(loader.Revision) {
			return errors.Errorf("%s.revision must be a lowercase 40-character commit hash", path)
		}
	case utils.DatasetLoaderJSON, utils.DatasetLoaderCSV, utils.DatasetLoaderParquet, utils.DatasetLoaderText:
		if !isHTTPDatasetSource(dataset.Source) {
			return errors.Errorf("%s type %s requires an absolute HTTP(S) source", path, loader.Type)
		}
		if loader.Subset != "" {
			return errors.Errorf("%s.subset is supported only for type huggingface", path)
		}
		if loader.Revision != "" {
			return errors.Errorf("%s.revision is supported only for type huggingface", path)
		}
		if loader.Checksum != "" && !datasetChecksumPattern.MatchString(loader.Checksum) {
			return errors.Errorf("%s.checksum must use lowercase sha256:<64 hex> format", path)
		}
	default:
		return errors.Errorf("%s.type %s is not supported", path, loader.Type)
	}

	return nil
}

func isHTTPDatasetSource(source string) bool {
	parsedSource, err := url.Parse(source)
	if err != nil {
		return false
	}

	scheme := strings.ToLower(parsedSource.Scheme)
	return (scheme == "http" || scheme == "https") && parsedSource.IsAbs() && parsedSource.Host != ""
}

func datasetReproducibilityWarnings(c *config.FineTuneConfig) []string {
	warnings := make([]string, 0, len(c.Datasets))
	for datasetIndex, dataset := range c.Datasets {
		if dataset.Loader == nil {
			if isHTTPDatasetSource(dataset.Source) {
				warnings = append(warnings, fmt.Sprintf("datasets[%d] remote JSON dataset has no checksum; its content is not reproducibly pinned", datasetIndex))
			} else {
				warnings = append(warnings, fmt.Sprintf("datasets[%d] Hugging Face dataset has no revision; its content is not reproducibly pinned", datasetIndex))
			}
			continue
		}

		switch dataset.Loader.Type {
		case utils.DatasetLoaderHuggingFace:
			if dataset.Loader.Revision == "" {
				warnings = append(warnings, fmt.Sprintf("datasets[%d] Hugging Face dataset has no revision; its content is not reproducibly pinned", datasetIndex))
			}
		case utils.DatasetLoaderJSON, utils.DatasetLoaderCSV, utils.DatasetLoaderParquet, utils.DatasetLoaderText:
			if dataset.Loader.Checksum == "" {
				warnings = append(warnings, fmt.Sprintf("datasets[%d] remote %s dataset has no checksum; its content is not reproducibly pinned", datasetIndex, dataset.Loader.Type))
			}
		}
	}

	return warnings
}

// isSupportedUnslothOptimizer limits OptimizerNames to dependencies in the frozen environment.
func isSupportedUnslothOptimizer(optimizer string) bool {
	switch optimizer {
	case "adamw_torch", "adamw_torch_fused", "adafactor", "adamw_torch_4bit", "adamw_torch_8bit", "ademamix", "sgd", "adagrad",
		"adamw_bnb_8bit", "adamw_8bit", "ademamix_8bit", "lion_8bit", "lion_32bit", "paged_adamw_32bit",
		"paged_adamw_8bit", "paged_ademamix_32bit", "paged_ademamix_8bit", "paged_lion_32bit", "paged_lion_8bit",
		"rmsprop", "rmsprop_bnb", "rmsprop_bnb_8bit", "rmsprop_bnb_32bit":
		return true
	default:
		return false
	}
}

// isSupportedUnslothScheduler limits SchedulerType to schedulers supported by the current API.
func isSupportedUnslothScheduler(scheduler string) bool {
	switch scheduler {
	case "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
		"inverse_sqrt":
		return true
	default:
		return false
	}
}

// isSupportedUnslothQuantization mirrors the quantization methods supported by the pinned Unsloth integration.
func isSupportedUnslothQuantization(quantization string) bool {
	switch strings.ToLower(quantization) {
	case "not_quantized", "fast_quantized", "quantized", "f32", "bf16", "f16", "q8_0", "q4_k_m", "q5_k_m",
		"q2_k", "q2_k_l", "q3_k_l", "q3_k_m", "q3_k_s", "q4_0", "q4_1", "q4_k_s", "q4_k", "q5_k",
		"q5_0", "q5_1", "q5_k_s", "q6_k", "q3_k_xs":
		return true
	default:
		return false
	}
}

func isPathSafeOutputName(name string) bool {
	if name == "" || name == "." || name == ".." {
		return false
	}

	for _, character := range name {
		if character >= 'a' && character <= 'z' ||
			character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' ||
			character == '.' || character == '-' || character == '_' {
			continue
		}
		return false
	}

	return true
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

	if slices.Contains(c.Backends, utils.BackendVLLM) && c.Runtime != utils.RuntimeNVIDIA {
		return errors.New("vllm backend only supports nvidia cuda runtime. please add 'runtime: cuda' to your aikitfile.yaml")
	}

	if c.Runtime == utils.RuntimeAppleSilicon && len(c.Backends) > 0 {
		for _, backend := range c.Backends {
			if backend != utils.BackendLlamaCpp {
				return errors.New("apple silicon runtime only supports llama-cpp backend")
			}
		}
	}

	// Runner mode (backends without models) is not supported on Apple Silicon
	// because the base image is Fedora-based and runner dependencies require apt-get.
	if c.Runtime == utils.RuntimeAppleSilicon && len(c.Backends) > 0 && len(c.Models) == 0 {
		return errors.New("runner mode (backends without models) is not supported on apple silicon runtime")
	}

	if c.Runtime == utils.RuntimeROCm && len(c.Backends) > 0 {
		for _, backend := range c.Backends {
			if backend != utils.BackendLlamaCpp {
				return errors.New("rocm runtime only supports llama-cpp backend")
			}
		}
	}

	backends := []string{utils.BackendLlamaCpp, utils.BackendDiffusers, utils.BackendVLLM}
	for _, b := range c.Backends {
		if !slices.Contains(backends, b) {
			return errors.Errorf("backend %s is not supported", b)
		}
	}

	runtimes := []string{"", utils.RuntimeNVIDIA, utils.RuntimeROCm, utils.RuntimeAppleSilicon}
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
