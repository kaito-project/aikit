package build

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	loadToMemoryTestModelName = "model"
	testHTTPSPrefix           = "https://"
	testMutableHubWarning     = "datasets[0] Hugging Face dataset has no revision; its content is not reproducibly pinned"
	testURLFragment           = "fragment"
	testURLToken              = "token"
	testURLValue              = "value"
	adapterQuantizeError      = "output.quantize cannot be configured when output.format is adapter"
)

func Test_validateConfig(t *testing.T) {
	type args struct {
		c *config.InferenceConfig
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name:    "no config",
			args:    args{c: &config.InferenceConfig{}},
			wantErr: true,
		},
		{
			name: "unsupported api version",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v10",
			}},
			wantErr: true,
		},
		{
			name: "invalid runtime",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1",
				Runtime:    "foo",
			}},
			wantErr: true,
		},
		{
			name: "valid backend",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"diffusers"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "catalog backend is deferred to platform resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"foo"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "backend runtime compatibility is deferred to catalog resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"diffusers"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "valid vllm backend with cuda runtime",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"vllm"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "vllm runtime compatibility is deferred to catalog resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"vllm"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "valid vllm-cpp backend with CPU runtime",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{utils.BackendVLLMCpp},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "valid vllm-cpp backend with cuda runtime",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeNVIDIA,
				Backends:   []string{utils.BackendVLLMCpp},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "vllm-cpp rocm compatibility is deferred to catalog resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeROCm,
				Backends:   []string{utils.BackendVLLMCpp},
			}},
			wantErr: false,
		},
		{
			name: "vllm-cpp apple compatibility is deferred to catalog resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeAppleSilicon,
				Backends:   []string{utils.BackendVLLMCpp},
			}},
			wantErr: false,
		},
		{
			name: "invalid backend name",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"exllama", "diffusers"},
				Models: []config.Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			}},
			wantErr: true,
		},
		{
			name: "valid runner mode - backends with no models (llama-cpp cpu)",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				Backends:     []string{utils.BackendLlamaCpp},
				LoadToMemory: []string{loadToMemoryTestModelName},
			}},
			wantErr: false,
		},
		{
			name: "valid runner mode - backends with no models (llama-cpp cuda)",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"llama-cpp"},
			}},
			wantErr: false,
		},
		{
			name: "valid runner mode - diffusers with cuda",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"diffusers"},
			}},
			wantErr: false,
		},
		{
			name: "valid runner mode - vllm with cuda",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "cuda",
				Backends:   []string{"vllm"},
			}},
			wantErr: false,
		},
		{
			name: "runner profile compatibility is deferred to catalog resolution",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "applesilicon",
				Backends:   []string{"llama-cpp"},
			}},
			wantErr: false,
		},
		{
			name: "loadToMemory rejects an empty model name",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{""},
			}},
			wantErr: true,
		},
		{
			name: "loadToMemory rejects a whitespace-only model name",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{" \t"},
			}},
			wantErr: true,
		},
		{
			name: "loadToMemory rejects a null character",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{"model\x00name"},
			}},
			wantErr: true,
		},
		{
			name: "loadToMemory rejects a comma",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{"chat,embeddings"},
			}},
			wantErr: true,
		},
		{
			name: "loadToMemory rejects a backslash",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{`model\`, "embeddings"},
			}},
			wantErr: true,
		},
		{
			name: "loadToMemory rejects duplicate model names",
			args: args{c: &config.InferenceConfig{
				APIVersion:   "v1alpha1",
				LoadToMemory: []string{loadToMemoryTestModelName, loadToMemoryTestModelName},
			}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateInferenceConfig(tt.args.c); (err != nil) != tt.wantErr {
				t.Errorf("validateConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestResolveBackendPlans(t *testing.T) {
	tests := []struct {
		name            string
		config          *config.InferenceConfig
		targetPlatforms []*specs.Platform
		wantErr         bool
	}{
		{
			name: "llama-cpp backend with arm64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: false,
		},
		{
			name: "diffusers backend without its required CUDA selector fails",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"diffusers"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: true,
		},
		{
			name: "mixed platforms with llama-cpp backend - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: false,
		},
		{
			name: "no backends specified with arm64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: false,
		},
		{
			name: "vllm backend without a CPU catalog tuple fails",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"vllm"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: true,
		},
		{
			name: "vllm-cpp CPU backend with amd64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			},
			wantErr: false,
		},
		{
			name: "vllm-cpp CPU backend with arm64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			},
			wantErr: false,
		},
		{
			name: "vllm-cpp CUDA backend with amd64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeNVIDIA,
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformAMD64, OS: utils.PlatformLinux},
			},
			wantErr: false,
		},
		{
			name: "vllm-cpp CUDA backend with arm64 platform - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeNVIDIA,
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformARM64, OS: utils.PlatformLinux},
			},
			wantErr: true,
		},
		{
			name: "vllm-cpp CPU backend with unsupported architecture - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "ppc64le", OS: utils.PlatformLinux},
			},
			wantErr: true,
		},
		{
			name: "vllm-cpp CPU backend with darwin platform - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformARM64, OS: "darwin"},
			},
			wantErr: true,
		},
		{
			name: "vllm-cpp CUDA backend with windows platform - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    utils.RuntimeNVIDIA,
				Backends:   []string{utils.BackendVLLMCpp},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: utils.PlatformAMD64, OS: "windows"},
			},
			wantErr: true,
		},
		{
			name: "missing ROCm tuple fails closed",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "rocm",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
			},
			wantErr: true,
		},
		{
			name: "rocm runtime with arm64 platform - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "rocm",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: true,
		},
		{
			name: "rocm runtime with mixed platforms - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "rocm",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plans, err := resolveBackendPlans(tt.config, tt.targetPlatforms)
			if (err != nil) != tt.wantErr {
				t.Errorf("resolveBackendPlans() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err == nil && len(plans) != len(tt.targetPlatforms) {
				t.Errorf("resolveBackendPlans() plans = %d, want %d", len(plans), len(tt.targetPlatforms))
			}
		})
	}
}

func Test_validateFineTuneConfig(t *testing.T) {
	const (
		invalidOutputNameError = "output name must be a safe filename containing only letters, numbers, dots, hyphens, or underscores"
		testMessagesSource     = "messages"
		testTextSource         = "text"
	)

	tests := []struct {
		name      string
		mutate    func(*config.FineTuneConfig)
		nilConfig bool
		wantErr   string
	}{
		{name: "valid alpaca dataset type"},
		{
			name: "valid messages dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetMessages
			},
		},
		{
			name: "valid sharegpt dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetShareGPT
			},
		},
		{
			name: "valid messages response loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetMessages
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
		},
		{
			name: "valid sharegpt response loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetShareGPT
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
		},
		{
			name: "valid prompt-completion dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetPromptCompletion
			},
		},
		{
			name: "valid text dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetText
			},
		},
		{name: "nil config", nilConfig: true, wantErr: "fine-tune config is not defined"},
		{
			name: "missing api version",
			mutate: func(c *config.FineTuneConfig) {
				c.APIVersion = ""
			},
			wantErr: "apiVersion is not defined",
		},
		{
			name: "unsupported api version",
			mutate: func(c *config.FineTuneConfig) {
				c.APIVersion = "v10"
			},
			wantErr: "apiVersion v10 is not supported",
		},
		{
			name: "unsupported target",
			mutate: func(c *config.FineTuneConfig) {
				c.Target = "other"
			},
			wantErr: "target other is not supported",
		},
		{
			name: "missing base model",
			mutate: func(c *config.FineTuneConfig) {
				c.BaseModel = "  "
			},
			wantErr: "baseModel is not defined",
		},
		{
			name: "no datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = nil
			},
			wantErr: "no datasets defined",
		},
		{
			name: "valid multiple full-sequence datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = []config.Dataset{
					{Source: "alpaca", Type: utils.DatasetAlpaca},
					{Source: testTextSource, Type: utils.DatasetText},
					{Source: testMessagesSource, Type: utils.DatasetMessages},
					{Source: "sharegpt", Type: utils.DatasetShareGPT},
				}
			},
		},
		{
			name: "valid multiple prompt-completion datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = []config.Dataset{
					{Source: "first", Type: utils.DatasetPromptCompletion},
					{Source: "second", Type: utils.DatasetPromptCompletion},
				}
			},
		},
		{
			name: "valid multiple response-only chat datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = []config.Dataset{
					{Source: testMessagesSource, Type: utils.DatasetMessages},
					{Source: "sharegpt", Type: utils.DatasetShareGPT},
				}
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
		},
		{
			name: "incompatible later prompt-completion dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, config.Dataset{Source: "other", Type: utils.DatasetPromptCompletion})
			},
			wantErr: "datasets[1] type prompt-completion is incompatible with datasets[0] type alpaca: completion-only and full-sequence datasets cannot be combined",
		},
		{
			name: "incompatible later full-sequence dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = []config.Dataset{
					{Source: "first", Type: utils.DatasetPromptCompletion},
					{Source: "second", Type: utils.DatasetText},
				}
			},
			wantErr: "datasets[1] type text is incompatible with datasets[0] type prompt-completion: full-sequence and completion-only datasets cannot be combined",
		},
		{
			name: "missing dataset source",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Source = "\t"
			},
			wantErr: "datasets[0].source is not defined",
		},
		{
			name: "missing later dataset source",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, config.Dataset{Source: " ", Type: utils.DatasetText})
			},
			wantErr: "datasets[1].source is not defined",
		},
		{
			name: "unsupported dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = "other"
			},
			wantErr: "datasets[0].type other is not supported",
		},
		{
			name: "unsupported later dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, config.Dataset{Source: "other", Type: "preference"})
			},
			wantErr: "datasets[1].type preference is not supported",
		},
		{
			name: "invalid later dataset loader",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, config.Dataset{
					Source: "organization/text",
					Type:   utils.DatasetText,
					Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace},
				})
			},
			wantErr: "datasets[1].loader.split is not defined",
		},
		{
			name: "missing loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Loss = ""
			},
			wantErr: "config.unsloth.loss is not defined",
		},
		{
			name: "whitespace loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Loss = " \t"
			},
			wantErr: "config.unsloth.loss is not defined",
		},
		{
			name: "unsupported loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Loss = "assistant"
			},
			wantErr: "config.unsloth.loss assistant is not supported",
		},
		{
			name: "response loss with alpaca dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
			wantErr: "datasets[0] type alpaca: config.unsloth.loss response is supported only for messages and sharegpt datasets",
		},
		{
			name: "response loss with prompt-completion dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetPromptCompletion
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
			wantErr: "datasets[0] type prompt-completion: config.unsloth.loss response is supported only for messages and sharegpt datasets",
		},
		{
			name: "response loss with text dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetText
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
			wantErr: "datasets[0] type text: config.unsloth.loss response is supported only for messages and sharegpt datasets",
		},
		{
			name: "response loss with later non-chat dataset",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = []config.Dataset{
					{Source: testMessagesSource, Type: utils.DatasetMessages},
					{Source: testTextSource, Type: utils.DatasetText},
				}
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
			wantErr: "datasets[1] type text: config.unsloth.loss response is supported only for messages and sharegpt datasets",
		},
		{
			name: "response loss with packing",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetMessages
				c.Config.Unsloth.Loss = utils.SFTLossResponse
				c.Config.Unsloth.Packing = true
			},
			wantErr: "config.unsloth.loss response does not support packing because response masks must not cross conversation boundaries",
		},
		{
			name: "non-positive max sequence length",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.MaxSeqLength = 0
			},
			wantErr: "config.unsloth.maxSeqLength must be greater than zero",
		},
		{
			name: "non-positive batch size",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.BatchSize = 0
			},
			wantErr: "config.unsloth.batchSize must be greater than zero",
		},
		{
			name: "non-positive gradient accumulation steps",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.GradientAccumulationSteps = 0
			},
			wantErr: "config.unsloth.gradientAccumulationSteps must be greater than zero",
		},
		{
			name: "negative warmup steps",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.WarmupSteps = -1
			},
			wantErr: "config.unsloth.warmupSteps must be zero or greater",
		},
		{
			name: "non-positive max steps",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.MaxSteps = 0
			},
			wantErr: "config.unsloth.maxSteps must be greater than zero",
		},
		{
			name: "non-positive learning rate",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.LearningRate = 0
			},
			wantErr: "config.unsloth.learningRate must be a finite value greater than zero",
		},
		{
			name: "non-finite learning rate",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.LearningRate = math.NaN()
			},
			wantErr: "config.unsloth.learningRate must be a finite value greater than zero",
		},
		{
			name: "non-positive logging steps",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.LoggingSteps = 0
			},
			wantErr: "config.unsloth.loggingSteps must be greater than zero",
		},
		{
			name: "missing optimizer",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Optimizer = " "
			},
			wantErr: "config.unsloth.optimizer is not defined",
		},
		{
			name: "unsupported optimizer",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Optimizer = "adamw_8bti"
			},
			wantErr: "config.unsloth.optimizer adamw_8bti is not supported",
		},
		{
			name: "negative weight decay",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.WeightDecay = -0.1
			},
			wantErr: "config.unsloth.weightDecay must be a finite value zero or greater",
		},
		{
			name: "non-finite weight decay",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.WeightDecay = math.Inf(1)
			},
			wantErr: "config.unsloth.weightDecay must be a finite value zero or greater",
		},
		{
			name: "missing scheduler",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.LrSchedulerType = ""
			},
			wantErr: "config.unsloth.lrSchedulerType is not defined",
		},
		{
			name: "unsupported scheduler",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.LrSchedulerType = "linera"
			},
			wantErr: "config.unsloth.lrSchedulerType linera is not supported",
		},
		{
			name: "negative seed",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Seed = -1
			},
			wantErr: "config.unsloth.seed must be zero or greater",
		},
		{
			name: "missing output format",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Format = ""
			},
			wantErr: "output.format is not defined",
		},
		{
			name: "whitespace output format",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Format = " \t"
			},
			wantErr: "output.format is not defined",
		},
		{
			name: "unsupported output format",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Format = "safetensors"
			},
			wantErr: `output.format "safetensors" is not supported`,
		},
		{
			name: "missing quantization",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Quantize = ""
			},
			wantErr: "output.quantize is not defined",
		},
		{
			name: "whitespace quantization",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Quantize = " \t"
			},
			wantErr: "output.quantize is not defined",
		},
		{
			name: "unsupported quantization",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Quantize = "q1_unsupported"
			},
			wantErr: `output.quantize "q1_unsupported" is not supported`,
		},
		{
			name: "empty output name",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Name = ""
			},
			wantErr: invalidOutputNameError,
		},
		{
			name: "traversal output name",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Name = "../model"
			},
			wantErr: invalidOutputNameError,
		},
		{
			name: "nested output name",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Name = "directory/model"
			},
			wantErr: invalidOutputNameError,
		},
		{
			name: "windows path output name",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Name = `directory\model`
			},
			wantErr: invalidOutputNameError,
		},
		{
			name: "space in output name",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Name = "model name"
			},
			wantErr: invalidOutputNameError,
		},
		{
			name: "valid adapter output",
			mutate: func(c *config.FineTuneConfig) {
				c.Output.Format = config.FineTuneOutputFormatAdapter
			},
		},
		{
			name: "valid explicit zero values and uppercase output settings",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.WarmupSteps = 0
				c.Config.Unsloth.WeightDecay = 0
				c.Config.Unsloth.Seed = 0
				c.Output.Format = "GGUF"
				c.Output.Quantize = "Q4_K_M"
				c.Output.Name = "model.v1-test_2"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fineTuneConfig := validFineTuneConfig()
			if tt.nilConfig {
				fineTuneConfig = nil
			} else if tt.mutate != nil {
				tt.mutate(fineTuneConfig)
			}

			err := validateFinetuneConfig(fineTuneConfig)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateFinetuneConfig() error = %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateFinetuneConfig() error = nil, want %q", tt.wantErr)
			}
			if err.Error() != tt.wantErr {
				t.Errorf("validateFinetuneConfig() error = %q, want %q", err, tt.wantErr)
			}
		})
	}
}

func Test_validateFineTuneConfigRejectsDatasetSourceWhitespace(t *testing.T) {
	const (
		hubID                     = "organization/private-dataset"
		sourceWhitespaceTestSplit = "train"
	)
	secretURL := strings.Join([]string{
		testHTTPSPrefix, "whitespace-user", ":", "whitespace-credential",
		"@example.test/train.jsonl?", testURLToken, "=whitespace-value",
		"#whitespace-fragment",
	}, "")

	tests := []struct {
		name            string
		datasetIndex    int
		dataset         config.Dataset
		dpo             bool
		wantErr         string
		sensitiveValues []string
	}{
		{
			name:         "leading whitespace in credentialed HTTP URL",
			datasetIndex: 0,
			dataset: config.Dataset{
				Source: " " + secretURL,
				Type:   utils.DatasetPreference,
				Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: sourceWhitespaceTestSplit},
			},
			dpo:             true,
			wantErr:         "datasets[0].source must not contain leading or trailing whitespace",
			sensitiveValues: []string{secretURL, "whitespace-user", "whitespace-credential", "whitespace-value", "whitespace-fragment"},
		},
		{
			name:         "trailing whitespace in credentialed HTTP URL on later dataset",
			datasetIndex: 1,
			dataset: config.Dataset{
				Source: secretURL + "\t",
				Type:   utils.DatasetText,
				Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: sourceWhitespaceTestSplit},
			},
			wantErr:         "datasets[1].source must not contain leading or trailing whitespace",
			sensitiveValues: []string{secretURL, "whitespace-user", "whitespace-credential", "whitespace-value", "whitespace-fragment"},
		},
		{
			name:         "leading whitespace in Hugging Face ID",
			datasetIndex: 0,
			dataset: config.Dataset{
				Source: "\n" + hubID,
				Type:   utils.DatasetAlpaca,
				Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: sourceWhitespaceTestSplit},
			},
			wantErr:         "datasets[0].source must not contain leading or trailing whitespace",
			sensitiveValues: []string{hubID},
		},
		{
			name:         "trailing Unicode whitespace in Hugging Face ID on later dataset",
			datasetIndex: 1,
			dataset: config.Dataset{
				Source: hubID + "\u00a0",
				Type:   utils.DatasetText,
				Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: sourceWhitespaceTestSplit},
			},
			wantErr:         "datasets[1].source must not contain leading or trailing whitespace",
			sensitiveValues: []string{hubID},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fineTuneConfig := validFineTuneConfig()
			if tt.dpo {
				fineTuneConfig = validDPOFineTuneConfig()
			}
			if tt.datasetIndex == 0 {
				fineTuneConfig.Datasets[0] = tt.dataset
			} else {
				fineTuneConfig.Datasets = append(fineTuneConfig.Datasets, tt.dataset)
			}

			err := validateFinetuneConfig(fineTuneConfig)
			if err == nil {
				t.Fatalf("validateFinetuneConfig() error = nil, want %q", tt.wantErr)
			}
			if err.Error() != tt.wantErr {
				t.Fatalf("validateFinetuneConfig() error = %q, want %q", err, tt.wantErr)
			}
			for _, sensitiveValue := range tt.sensitiveValues {
				if strings.Contains(err.Error(), sensitiveValue) {
					t.Fatalf("validation error leaked dataset source detail %q: %q", sensitiveValue, err)
				}
			}
		})
	}
}

func Test_validateDPOFineTuneConfig(t *testing.T) {
	const (
		revision       = "0123456789abcdef0123456789abcdef01234567"
		checksum       = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
		dpoLoaderSplit = "train"
		betaError      = "objective.beta must be a finite value greater than zero"
	)

	tests := []struct {
		name    string
		mutate  func(*config.FineTuneConfig)
		wantErr string
	}{
		{name: "valid defaults"},
		{
			name: "DPO rejects multiple preference datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, c.Datasets[0])
			},
			wantErr: "objective type dpo requires exactly one dataset",
		},
		{
			name: "valid pinned Hugging Face loader",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Loader = &config.DatasetLoaderSpec{
					Type: utils.DatasetLoaderHuggingFace, Split: dpoLoaderSplit, Revision: revision,
				}
			},
		},
		{
			name: "valid checksummed JSON loader",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Source = "https://example.test/preferences.jsonl"
				c.Datasets[0].Loader = &config.DatasetLoaderSpec{
					Type: utils.DatasetLoaderJSON, Split: dpoLoaderSplit, Checksum: checksum,
				}
			},
		},
		{
			name: "missing objective type",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Type = ""
			},
			wantErr: "objective.type is not defined",
		},
		{
			name: "whitespace objective type",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Type = " \t"
			},
			wantErr: "objective.type is not defined",
		},
		{
			name: "unsupported objective type",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Type = "orpo"
			},
			wantErr: "objective.type orpo is not supported",
		},
		{
			name: "SFT rejects preference",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective = config.FineTuneObjectiveSpec{Type: utils.ObjectiveSFT}
			},
			wantErr: "dataset type preference is supported only for objective type dpo",
		},
		{
			name: "SFT rejects DPO settings",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective = config.FineTuneObjectiveSpec{Type: utils.ObjectiveSFT, Beta: 0.1}
				c.Datasets[0].Type = utils.DatasetAlpaca
			},
			wantErr: "objective beta, lossType, and maxPromptLength are supported only for objective type dpo",
		},
		{
			name: "DPO rejects Alpaca",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetAlpaca
			},
			wantErr: "objective type dpo requires dataset type preference, got alpaca",
		},
		{
			name: "DPO rejects messages",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetMessages
			},
			wantErr: "objective type dpo requires dataset type preference, got messages",
		},
		{
			name: "DPO rejects ShareGPT",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetShareGPT
			},
			wantErr: "objective type dpo requires dataset type preference, got sharegpt",
		},
		{
			name: "DPO rejects prompt-completion",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetPromptCompletion
			},
			wantErr: "objective type dpo requires dataset type preference, got prompt-completion",
		},
		{
			name: "DPO rejects text",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = utils.DatasetText
			},
			wantErr: "objective type dpo requires dataset type preference, got text",
		},
		{
			name: "DPO rejects text loader",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Source = "https://example.test/preferences.txt"
				c.Datasets[0].Loader = &config.DatasetLoaderSpec{Type: utils.DatasetLoaderText, Split: "train"}
			},
			wantErr: "dataset type preference does not support loader type text",
		},
		{
			name: "DPO rejects response loss",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Loss = utils.SFTLossResponse
			},
			wantErr: "config.unsloth.loss response is an SFT-only setting and is not supported for objective type dpo",
		},
		{
			name: "DPO rejects packing",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.Packing = true
			},
			wantErr: "objective type dpo does not support config.unsloth.packing",
		},
		{
			name: "zero beta",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Beta = 0
			},
			wantErr: betaError,
		},
		{
			name: "negative beta",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Beta = -0.1
			},
			wantErr: betaError,
		},
		{
			name: "NaN beta",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Beta = math.NaN()
			},
			wantErr: betaError,
		},
		{
			name: "positive infinity beta",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Beta = math.Inf(1)
			},
			wantErr: betaError,
		},
		{
			name: "negative infinity beta",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.Beta = math.Inf(-1)
			},
			wantErr: betaError,
		},
		{
			name: "missing loss type",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.LossType = ""
			},
			wantErr: "objective.lossType is not defined",
		},
		{
			name: "unsupported loss type",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.LossType = "hinge"
			},
			wantErr: "objective.lossType hinge is not supported",
		},
		{
			name: "zero max prompt length",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.MaxPromptLength = 0
			},
			wantErr: "objective.maxPromptLength must be greater than zero",
		},
		{
			name: "negative max prompt length",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.MaxPromptLength = -1
			},
			wantErr: "objective.maxPromptLength must be greater than zero",
		},
		{
			name: "max prompt length exceeds sequence length",
			mutate: func(c *config.FineTuneConfig) {
				c.Objective.MaxPromptLength = c.Config.Unsloth.MaxSeqLength + 1
			},
			wantErr: "objective.maxPromptLength must not exceed config.unsloth.maxSeqLength",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fineTuneConfig := validDPOFineTuneConfig()
			if tt.mutate != nil {
				tt.mutate(fineTuneConfig)
			}
			err := validateFinetuneConfig(fineTuneConfig)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateFinetuneConfig() error = %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateFinetuneConfig() error = nil, want %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateFinetuneConfig() error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func Test_validateNormalizedFineTuneConfig(t *testing.T) {
	datasetTypes := []string{utils.DatasetAlpaca, utils.DatasetMessages, utils.DatasetPromptCompletion, utils.DatasetShareGPT, utils.DatasetText}
	for _, datasetType := range datasetTypes {
		t.Run(datasetType, func(t *testing.T) {
			_, fineTuneConfig, err := config.NewFromBytes([]byte(`
apiVersion: v1alpha1
baseModel: unsloth/test-model
datasets:
  - source: test-dataset
    type: ` + datasetType + "\n"))
			if err != nil {
				t.Fatalf("config.NewFromBytes() error = %v", err)
			}
			if fineTuneConfig == nil {
				t.Fatal("config.NewFromBytes() returned no fine-tune config")
			}
			fineTuneConfig.Target = utils.TargetUnsloth

			if got := fineTuneConfig.Datasets[0].Type; got != datasetType {
				t.Fatalf("dataset type = %q, want %q", got, datasetType)
			}
			if got := fineTuneConfig.Config.Unsloth.Loss; got != utils.SFTLossAll {
				t.Fatalf("loss = %q, want default %q", got, utils.SFTLossAll)
			}
			if err := validateFinetuneConfig(fineTuneConfig); err != nil {
				t.Fatalf("validateFinetuneConfig() error = %v", err)
			}
		})
	}
}

func Test_validateNormalizedDPOFineTuneConfig(t *testing.T) {
	_, fineTuneConfig, err := config.NewFromBytes([]byte(`
apiVersion: v1alpha1
baseModel: unsloth/test-model
objective:
  type: dpo
datasets:
  - source: organization/preferences
    type: preference
config:
  unsloth:
    packing: false
`))
	if err != nil {
		t.Fatalf("config.NewFromBytes() error = %v", err)
	}
	if fineTuneConfig == nil {
		t.Fatal("config.NewFromBytes() returned no fine-tune config")
	}
	fineTuneConfig.Target = utils.TargetUnsloth

	if got := fineTuneConfig.Objective; got != (config.FineTuneObjectiveSpec{
		Type:            utils.ObjectiveDPO,
		Beta:            0.1,
		LossType:        utils.DPOLossSigmoid,
		MaxPromptLength: 512,
	}) {
		t.Fatalf("objective = %#v, want normalized DPO defaults", got)
	}
	if got := fineTuneConfig.Config.Unsloth.LearningRate; got != 0.000001 {
		t.Fatalf("learning rate = %g, want DPO default 0.000001", got)
	}
	if err := validateFinetuneConfig(fineTuneConfig); err != nil {
		t.Fatalf("validateFinetuneConfig() error = %v", err)
	}
}

func Test_validateFineTuneConfigNormalizesQuantization(t *testing.T) {
	config := validFineTuneConfig()
	config.Output.Format = "GGUF"
	config.Output.Quantize = "Q4_K_M"

	if err := validateFinetuneConfig(config); err != nil {
		t.Fatalf("validateFinetuneConfig() error = %v", err)
	}
	if config.Output.Quantize != "q4_k_m" {
		t.Fatalf("normalized quantization = %q, want q4_k_m", config.Output.Quantize)
	}
	if config.Output.Format != "gguf" {
		t.Fatalf("normalized output format = %q, want gguf", config.Output.Format)
	}
}

func Test_validateFineTuneAdapterRejectsExplicitQuantize(t *testing.T) {
	tests := []struct {
		name       string
		defaults   string
		outputBody string
		wantErr    string
	}{
		{name: "omitted"},
		{name: "value", outputBody: "  quantize: q4_k_m\n", wantErr: adapterQuantizeError},
		{name: "empty", outputBody: "  quantize: \"\"\n", wantErr: adapterQuantizeError},
		{name: "null", outputBody: "  quantize: null\n", wantErr: adapterQuantizeError},
		{
			name:       "inherited value",
			defaults:   "outputDefaults: &outputDefaults\n  quantize: q8_0\n",
			outputBody: "  <<: *outputDefaults\n",
			wantErr:    adapterQuantizeError,
		},
		{
			name:       "inherited null",
			defaults:   "outputDefaults: &outputDefaults\n  quantize: null\n",
			outputBody: "  <<: *outputDefaults\n",
			wantErr:    adapterQuantizeError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := "apiVersion: v1alpha1\n" +
				"baseModel: test-model\n" +
				"datasets:\n" +
				"  - source: test-dataset\n" +
				"    type: alpaca\n" +
				tt.defaults +
				"output:\n" +
				"  format: ADAPTER\n" +
				tt.outputBody
			_, fineTuneConfig, err := config.NewFromBytes([]byte(input))
			if err != nil {
				t.Fatalf("config.NewFromBytes() error = %v", err)
			}
			fineTuneConfig.Target = utils.TargetUnsloth

			err = validateFinetuneConfig(fineTuneConfig)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateFinetuneConfig() error = %v", err)
				}
				if fineTuneConfig.Output.Format != config.FineTuneOutputFormatAdapter {
					t.Fatalf("normalized output format = %q, want adapter", fineTuneConfig.Output.Format)
				}
				return
			}
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("validateFinetuneConfig() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func Test_isSupportedUnslothOptimizer(t *testing.T) {
	supported := []string{
		"adamw_torch", "adamw_torch_fused", "adafactor", "adamw_torch_4bit", "adamw_torch_8bit", "ademamix", "sgd", "adagrad",
		"adamw_bnb_8bit", "adamw_8bit", "ademamix_8bit", "lion_8bit", "lion_32bit", "paged_adamw_32bit",
		"paged_adamw_8bit", "paged_ademamix_32bit", "paged_ademamix_8bit", "paged_lion_32bit", "paged_lion_8bit",
		"rmsprop", "rmsprop_bnb", "rmsprop_bnb_8bit", "rmsprop_bnb_32bit",
	}
	for _, optimizer := range supported {
		t.Run(optimizer, func(t *testing.T) {
			if !isSupportedUnslothOptimizer(optimizer) {
				t.Errorf("isSupportedUnslothOptimizer(%q) = false, want true", optimizer)
			}
		})
	}

	unsupported := []string{
		"unsupported", "adamw_torch_xla", "adamw_torch_npu_fused", "adamw_apex_fused", "adamw_anyprecision",
		"galore_adamw", "galore_adamw_8bit", "galore_adafactor", "galore_adamw_layerwise",
		"galore_adamw_8bit_layerwise", "galore_adafactor_layerwise", "lomo", "adalomo", "grokadamw",
		"schedule_free_radam", "schedule_free_adamw", "schedule_free_sgd", "apollo_adamw",
		"apollo_adamw_layerwise", "stable_adamw",
	}
	for _, optimizer := range unsupported {
		if isSupportedUnslothOptimizer(optimizer) {
			t.Errorf("isSupportedUnslothOptimizer(%q) = true, want false", optimizer)
		}
	}
}

func Test_isSupportedUnslothScheduler(t *testing.T) {
	supported := []string{
		"linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
		"inverse_sqrt",
	}
	for _, scheduler := range supported {
		t.Run(scheduler, func(t *testing.T) {
			if !isSupportedUnslothScheduler(scheduler) {
				t.Errorf("isSupportedUnslothScheduler(%q) = false, want true", scheduler)
			}
		})
	}

	for _, scheduler := range []string{
		"unsupported", "cosine_with_min_lr", "cosine_warmup_with_min_lr", "reduce_lr_on_plateau",
		"warmup_stable_decay", "greedy",
	} {
		if isSupportedUnslothScheduler(scheduler) {
			t.Errorf("isSupportedUnslothScheduler(%q) = true, want false", scheduler)
		}
	}
}

func Test_isSupportedUnslothQuantization(t *testing.T) {
	supported := []string{
		"not_quantized", "fast_quantized", "quantized", "f32", "bf16", "f16", "q8_0", "q4_k_m", "q5_k_m",
		"q2_k", "q2_k_l", "q3_k_l", "q3_k_m", "q3_k_s", "q4_0", "q4_1", "q4_k_s", "q4_k", "q5_k",
		"q5_0", "q5_1", "q5_k_s", "q6_k", "q3_k_xs",
	}
	for _, quantization := range supported {
		t.Run(quantization, func(t *testing.T) {
			if !isSupportedUnslothQuantization(quantization) {
				t.Errorf("isSupportedUnslothQuantization(%q) = false, want true", quantization)
			}
		})
	}

	if isSupportedUnslothQuantization("unsupported") {
		t.Error("isSupportedUnslothQuantization(\"unsupported\") = true, want false")
	}
	if !isSupportedUnslothQuantization("Q4_K_M") {
		t.Error("isSupportedUnslothQuantization(\"Q4_K_M\") = false, want true")
	}
}

func validFineTuneConfig() *config.FineTuneConfig {
	return &config.FineTuneConfig{
		APIVersion: "v1alpha1",
		Target:     "unsloth",
		BaseModel:  "unsloth/test-model",
		Objective:  config.FineTuneObjectiveSpec{Type: utils.ObjectiveSFT},
		Datasets: []config.Dataset{
			{Source: "test-dataset", Type: "alpaca"},
		},
		Config: config.FineTuneConfigSpec{
			Unsloth: config.FineTuneConfigUnslothSpec{
				MaxSeqLength:              2048,
				LoadIn4bit:                true,
				Loss:                      utils.SFTLossAll,
				BatchSize:                 2,
				GradientAccumulationSteps: 4,
				WarmupSteps:               10,
				MaxSteps:                  60,
				LearningRate:              0.0002,
				LoggingSteps:              1,
				Optimizer:                 "adamw_8bit",
				WeightDecay:               0.01,
				LrSchedulerType:           "linear",
				Seed:                      42,
			},
		},
		Output: config.FineTuneOutputSpec{
			Format:   config.FineTuneOutputFormatGGUF,
			Quantize: "q4_k_m",
			Name:     "aikit-model",
		},
	}
}

func validDPOFineTuneConfig() *config.FineTuneConfig {
	c := validFineTuneConfig()
	c.Objective = config.FineTuneObjectiveSpec{
		Type:            utils.ObjectiveDPO,
		Beta:            0.1,
		LossType:        utils.DPOLossSigmoid,
		MaxPromptLength: 512,
	}
	c.Datasets[0].Type = utils.DatasetPreference
	c.Config.Unsloth.LearningRate = 0.000001
	return c
}

func Test_parseFineTuneBuildOptions(t *testing.T) {
	tests := []struct {
		name        string
		opts        map[string]string
		wantVersion string
		wantDevice  string
		wantErr     bool
	}{
		{name: "omitted"},
		{name: "valid", opts: map[string]string{"build-arg:nvidiaDriverVersion": "590.48.01"}, wantVersion: "590.48.01"},
		{name: "trimmed", opts: map[string]string{"build-arg:nvidiaDriverVersion": " 590.48.01 "}, wantVersion: "590.48.01"},
		{name: "two-component WSL version", opts: map[string]string{"build-arg:nvidiaDriverVersion": "572.83"}, wantVersion: "572.83"},
		{name: "missing minor", opts: map[string]string{"build-arg:nvidiaDriverVersion": "590"}, wantErr: true},
		{name: "shell input", opts: map[string]string{"build-arg:nvidiaDriverVersion": "$(id)"}, wantErr: true},
		{name: "CDI index", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=0"}, wantDevice: "nvidia.com/gpu=0"},
		{name: "CDI UUID", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=GPU-4f684ff2-f5d1-8b33-decf-42fac828778c"}, wantDevice: "nvidia.com/gpu=GPU-4f684ff2-f5d1-8b33-decf-42fac828778c"},
		{name: "CDI MIG UUID", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=MIG-5f9d9b6a-98d1-4f6a-9b49-4a2d4a651369"}, wantDevice: "nvidia.com/gpu=MIG-5f9d9b6a-98d1-4f6a-9b49-4a2d4a651369"},
		{name: "CDI on demand", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu"}, wantDevice: "nvidia.com/gpu"},
		{name: "CDI type index GPU", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=gpu0"}, wantDevice: "nvidia.com/gpu=gpu0"},
		{name: "CDI type index MIG", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=mig1:0"}, wantDevice: "nvidia.com/gpu=mig1:0"},
		{name: "short GPU alias", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=GPU-deadbeef"}, wantErr: true},
		{name: "custom MIG alias", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=MIG-custom-alias"}, wantErr: true},
		{name: "GPU UUID suffix", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=GPU-4f684ff2-f5d1-8b33-decf-42fac828778c-extra"}, wantErr: true},
		{name: "non-NVIDIA CDI device", opts: map[string]string{"build-arg:cdiDevice": "vendor.example/gpu=0"}, wantErr: true},
		{name: "CDI shell input", opts: map[string]string{"build-arg:cdiDevice": "nvidia.com/gpu=$(id)"}, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options, err := parseFineTuneBuildOptions(tt.opts)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseFineTuneBuildOptions() error = %v, wantErr %t", err, tt.wantErr)
			}
			if options.NVIDIADriverVersion != tt.wantVersion {
				t.Fatalf("driver version = %q, want %q", options.NVIDIADriverVersion, tt.wantVersion)
			}
			if options.CDIDevice != tt.wantDevice {
				t.Fatalf("CDI device = %q, want %q", options.CDIDevice, tt.wantDevice)
			}
		})
	}
}

func Test_validateDatasetLoader(t *testing.T) {
	const (
		testHubDatasetSource = "organization/dataset"
		testRemoteJSONSource = "https://example.test/train.json"
		testLoaderSplit      = "train"
		revision             = "0123456789abcdef0123456789abcdef01234567"
		checksum             = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
		revisionError        = "revision must be a lowercase 40-character commit hash"
		checksumError        = "checksum must use lowercase sha256:<64 hex> format"
	)

	tests := []struct {
		name    string
		source  string
		loader  *config.DatasetLoaderSpec
		wantErr string
	}{
		{name: "omitted loader", source: testHubDatasetSource},
		{
			name:   "pinned huggingface",
			source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Subset: "default", Split: "train_sft", Revision: revision},
		},
		{
			name:   "mutable huggingface",
			source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit},
		},
		{
			name:   "json",
			source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: "validation", Checksum: checksum},
		},
		{
			name:   "csv",
			source: "http://example.test/train.csv",
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderCSV, Split: testLoaderSplit, Checksum: checksum},
		},
		{
			name:   "parquet",
			source: "HTTPS://example.test/train.parquet",
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderParquet, Split: testLoaderSplit, Checksum: checksum},
		},
		{
			name:   "text",
			source: "https://example.test/train.txt",
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderText, Split: testLoaderSplit, Checksum: checksum},
		},
		{
			name: "missing type", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Split: testLoaderSplit}, wantErr: "datasets[0].loader.type is not defined",
		},
		{
			name: "unknown type", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: "arrow", Split: testLoaderSplit}, wantErr: "datasets[0].loader.type arrow is not supported",
		},
		{
			name: "missing split", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace}, wantErr: "datasets[0].loader.split is not defined",
		},
		{
			name: "split expression", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: "train[:10%]"}, wantErr: "datasets[0].loader.split must be a named split",
		},
		{
			name: "hyphenated split", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: "train-sft"}, wantErr: "datasets[0].loader.split must be a named split",
		},
		{
			name: "empty subset", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Subset: " \t"}, wantErr: "datasets[0].loader.subset must not be empty",
		},
		{
			name: "huggingface URL", source: strings.Join([]string{testHTTPSPrefix, "user-one", ":", "credential-one", "@example.test/data?token=", testURLValue, "#" + testURLFragment}, ""),
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit}, wantErr: "type huggingface does not support an HTTP(S) source",
		},
		{
			name: "huggingface checksum", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Checksum: checksum}, wantErr: "checksum is not supported for type huggingface",
		},
		{
			name: "branch revision", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Revision: "main"}, wantErr: revisionError,
		},
		{
			name: "short revision", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Revision: "01234567"}, wantErr: revisionError,
		},
		{
			name: "uppercase revision", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Revision: strings.ToUpper(revision)}, wantErr: revisionError,
		},
		{
			name: "remote loader non-URL", source: testHubDatasetSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit}, wantErr: "type json requires an absolute HTTP(S) source",
		},
		{
			name: "remote loader relative URL", source: "/train.json",
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit}, wantErr: "type json requires an absolute HTTP(S) source",
		},
		{
			name: "remote subset", source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit, Subset: "default"}, wantErr: "subset is supported only for type huggingface",
		},
		{
			name: "remote revision", source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit, Revision: revision}, wantErr: "revision is supported only for type huggingface",
		},
		{
			name: "checksum algorithm", source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit, Checksum: "md5:0123"}, wantErr: checksumError,
		},
		{
			name: "checksum length", source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit, Checksum: "sha256:0123"}, wantErr: checksumError,
		},
		{
			name: "uppercase checksum", source: testRemoteJSONSource,
			loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: testLoaderSplit, Checksum: strings.ToUpper(checksum)}, wantErr: checksumError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fineTuneConfig := validFineTuneConfig()
			fineTuneConfig.Datasets[0].Source = tt.source
			fineTuneConfig.Datasets[0].Loader = tt.loader
			err := validateFinetuneConfig(fineTuneConfig)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateFinetuneConfig() error = %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateFinetuneConfig() error = nil, want %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateFinetuneConfig() error = %q, want substring %q", err, tt.wantErr)
			}
			for _, secret := range []string{"user-one", "credential-one", testURLToken, testURLValue, testURLFragment} {
				if strings.Contains(err.Error(), secret) {
					t.Fatalf("validation error leaked URL detail %q: %q", secret, err)
				}
			}
		})
	}
}

func Test_datasetReproducibilityWarnings(t *testing.T) {
	const (
		privateDatasetSource = "organization/private-dataset"
		testLoaderSplit      = "train"
		revision             = "0123456789abcdef0123456789abcdef01234567"
		checksum             = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	)
	secretURL := strings.Join([]string{testHTTPSPrefix, "user-two", ":", "credential-two", "@example.test/train.parquet?token=", "private-value", "#" + testURLFragment}, "")

	tests := []struct {
		name         string
		dataset      config.Dataset
		wantWarnings []string
	}{
		{
			name:         "legacy hub",
			dataset:      config.Dataset{Source: privateDatasetSource, Type: utils.DatasetAlpaca},
			wantWarnings: []string{testMutableHubWarning},
		},
		{
			name:         "legacy URL",
			dataset:      config.Dataset{Source: secretURL, Type: utils.DatasetPromptCompletion},
			wantWarnings: []string{"datasets[0] remote JSON dataset has no checksum; its content is not reproducibly pinned"},
		},
		{
			name:         "mutable hub loader",
			dataset:      config.Dataset{Source: privateDatasetSource, Type: utils.DatasetMessages, Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit}},
			wantWarnings: []string{testMutableHubWarning},
		},
		{
			name:         "mutable remote loader",
			dataset:      config.Dataset{Source: secretURL, Type: utils.DatasetPromptCompletion, Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderParquet, Split: testLoaderSplit}},
			wantWarnings: []string{"datasets[0] remote parquet dataset has no checksum; its content is not reproducibly pinned"},
		},
		{
			name:         "pinned hub",
			dataset:      config.Dataset{Source: privateDatasetSource, Type: utils.DatasetMessages, Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: testLoaderSplit, Revision: revision}},
			wantWarnings: []string{},
		},
		{
			name:         "checksummed remote",
			dataset:      config.Dataset{Source: secretURL, Type: utils.DatasetPromptCompletion, Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderParquet, Split: testLoaderSplit, Checksum: checksum}},
			wantWarnings: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fineTuneConfig := validFineTuneConfig()
			fineTuneConfig.Datasets = []config.Dataset{tt.dataset}
			warnings := datasetReproducibilityWarnings(fineTuneConfig)
			if !reflect.DeepEqual(warnings, tt.wantWarnings) {
				t.Fatalf("warnings = %#v, want %#v", warnings, tt.wantWarnings)
			}
			for _, warning := range warnings {
				for _, secret := range []string{privateDatasetSource, "example.test", "user-two", "credential-two", testURLToken, "private-value", testURLFragment} {
					if strings.Contains(warning, secret) {
						t.Fatalf("warning leaked source detail %q: %q", secret, warning)
					}
				}
			}
		})
	}
}

func Test_datasetReproducibilityWarningsCoverEveryDataset(t *testing.T) {
	secretURL := strings.Join([]string{testHTTPSPrefix, "user", ":", "credential", "@example.test/second.json?token=", testURLValue, "#" + testURLFragment}, "")
	fineTuneConfig := validFineTuneConfig()
	fineTuneConfig.Datasets = []config.Dataset{
		{
			Source: "organization/first",
			Type:   utils.DatasetText,
			Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderHuggingFace, Split: "train"},
		},
		{
			Source: secretURL,
			Type:   utils.DatasetText,
			Loader: &config.DatasetLoaderSpec{Type: utils.DatasetLoaderJSON, Split: "train"},
		},
	}

	want := []string{
		testMutableHubWarning,
		"datasets[1] remote json dataset has no checksum; its content is not reproducibly pinned",
	}
	warnings := datasetReproducibilityWarnings(fineTuneConfig)
	if !reflect.DeepEqual(warnings, want) {
		t.Fatalf("warnings = %#v, want %#v", warnings, want)
	}
	for _, warning := range warnings {
		for _, secret := range []string{"organization/first", "example.test", "user", "credential", testURLToken, testURLValue, testURLFragment} {
			if strings.Contains(warning, secret) {
				t.Fatalf("warning leaked source detail %q: %q", secret, warning)
			}
		}
	}
}
