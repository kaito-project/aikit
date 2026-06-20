package config

import (
	"errors"
	"slices"

	"github.com/kaito-project/aikit/pkg/utils"
	pkgerrors "github.com/pkg/errors"
)

// Validate checks that the inference config is internally consistent and only
// references supported backends and runtimes. Membership errors (unknown
// backend / unknown runtime) are accumulated with errors.Join so that a config
// with several mistakes reports all of them at once; compatibility rules that
// only make sense for a known-good backend are checked afterwards.
func (c *InferenceConfig) Validate() error {
	if c.APIVersion == "" {
		return errors.New("apiVersion is not defined")
	}
	if c.APIVersion != utils.APIv1alpha1 {
		return pkgerrors.Errorf("apiVersion %s is not supported", c.APIVersion)
	}

	supportedBackends := []string{utils.BackendLlamaCpp, utils.BackendDiffusers, utils.BackendVLLM}
	supportedRuntimes := []string{"", utils.RuntimeNVIDIA, utils.RuntimeROCm, utils.RuntimeAppleSilicon}

	var membershipErrs []error
	for _, b := range c.Backends {
		if !slices.Contains(supportedBackends, b) {
			membershipErrs = append(membershipErrs, pkgerrors.Errorf("backend %s is not supported", b))
		}
	}
	if !slices.Contains(supportedRuntimes, c.Runtime) {
		membershipErrs = append(membershipErrs, pkgerrors.Errorf("runtime %s is not supported", c.Runtime))
	}
	if len(membershipErrs) > 0 {
		return errors.Join(membershipErrs...)
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

	return nil
}

// Validate checks that the finetune config is internally consistent and only
// references supported targets and dataset types.
func (c *FineTuneConfig) Validate() error {
	supportedFineTuneTargets := []string{utils.TargetUnsloth}

	if c.APIVersion == "" {
		return errors.New("apiVersion is not defined")
	}
	if c.APIVersion != utils.APIv1alpha1 {
		return pkgerrors.Errorf("apiVersion %s is not supported", c.APIVersion)
	}
	if !slices.Contains(supportedFineTuneTargets, c.Target) {
		return pkgerrors.Errorf("target %s is not supported", c.Target)
	}
	if len(c.Datasets) == 0 {
		return errors.New("no datasets defined")
	}
	if len(c.Datasets) > 1 {
		return errors.New("only one dataset is supported at this time")
	}
	for _, d := range c.Datasets {
		if d.Type != utils.DatasetAlpaca {
			return pkgerrors.Errorf("dataset type %s is not supported", d.Type)
		}
	}
	return nil
}

// FillDefaults populates unset finetune fields with their default values. When
// the target is unsloth, unsloth-specific hyperparameter defaults are applied
// as well.
func (c *FineTuneConfig) FillDefaults() {
	if c.Target == utils.TargetUnsloth {
		c.fillUnslothDefaults()
	}
	if c.Output.Quantize == "" {
		c.Output.Quantize = "q4_k_m"
	}
	if c.Output.Name == "" {
		c.Output.Name = "aikit-model"
	}
}

func (c *FineTuneConfig) fillUnslothDefaults() {
	u := &c.Config.Unsloth
	if u.MaxSeqLength == 0 {
		u.MaxSeqLength = 2048
	}
	if u.BatchSize == 0 {
		u.BatchSize = 2
	}
	if u.GradientAccumulationSteps == 0 {
		u.GradientAccumulationSteps = 4
	}
	if u.WarmupSteps == 0 {
		u.WarmupSteps = 10
	}
	if u.MaxSteps == 0 {
		u.MaxSteps = 60
	}
	if u.LearningRate == 0 {
		u.LearningRate = 0.0002
	}
	if u.LoggingSteps == 0 {
		u.LoggingSteps = 1
	}
	if u.Optimizer == "" {
		u.Optimizer = "adamw_8bit"
	}
	if u.WeightDecay == 0 {
		u.WeightDecay = 0.01
	}
	if u.LrSchedulerType == "" {
		u.LrSchedulerType = "linear"
	}
	if u.Seed == 0 {
		u.Seed = 42
	}
}
