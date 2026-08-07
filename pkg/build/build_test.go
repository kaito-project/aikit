package build

import (
	"math"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const loadToMemoryTestModelName = "model"

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
			name: "invalid backend",
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
			wantErr: true,
		},
		{
			name: "diffusers backend requires cuda runtime",
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
			wantErr: true,
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
			name: "vllm backend requires cuda runtime",
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
			wantErr: true,
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
			name: "runner mode not supported on apple silicon",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "applesilicon",
				Backends:   []string{"llama-cpp"},
			}},
			wantErr: true,
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

func Test_validateBackendPlatformCompatibility(t *testing.T) {
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
			name: "diffusers backend with arm64 platform - should fail",
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
			name: "vllm backend with arm64 platform - should fail",
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
			name: "rocm runtime with amd64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Runtime:    "rocm",
				Backends:   []string{"llama-cpp"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
			},
			wantErr: false,
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
			if err := validateBackendPlatformCompatibility(tt.config, tt.targetPlatforms); (err != nil) != tt.wantErr {
				t.Errorf("validateBackendPlatformCompatibility() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_validateFineTuneConfig(t *testing.T) {
	const invalidOutputNameError = "output name must be a safe filename containing only letters, numbers, dots, hyphens, or underscores"

	tests := []struct {
		name      string
		mutate    func(*config.FineTuneConfig)
		nilConfig bool
		wantErr   string
	}{
		{name: "valid"},
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
			name: "multiple datasets",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets = append(c.Datasets, config.Dataset{Source: "other", Type: "alpaca"})
			},
			wantErr: "only one dataset is supported at this time",
		},
		{
			name: "missing dataset source",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Source = "\t"
			},
			wantErr: "dataset source is not defined",
		},
		{
			name: "unsupported dataset type",
			mutate: func(c *config.FineTuneConfig) {
				c.Datasets[0].Type = "other"
			},
			wantErr: "dataset type other is not supported",
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
			name: "valid explicit zero values and uppercase quantization",
			mutate: func(c *config.FineTuneConfig) {
				c.Config.Unsloth.WarmupSteps = 0
				c.Config.Unsloth.WeightDecay = 0
				c.Config.Unsloth.Seed = 0
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

func Test_validateNormalizedFineTuneConfig(t *testing.T) {
	_, fineTuneConfig, err := config.NewFromBytes([]byte(`
apiVersion: v1alpha1
baseModel: unsloth/test-model
datasets:
  - source: test-dataset
    type: alpaca
`))
	if err != nil {
		t.Fatalf("config.NewFromBytes() error = %v", err)
	}
	if fineTuneConfig == nil {
		t.Fatal("config.NewFromBytes() returned no fine-tune config")
	}
	fineTuneConfig.Target = "unsloth"

	if err := validateFinetuneConfig(fineTuneConfig); err != nil {
		t.Fatalf("validateFinetuneConfig() error = %v", err)
	}
}

func Test_validateFineTuneConfigNormalizesQuantization(t *testing.T) {
	config := validFineTuneConfig()
	config.Output.Quantize = "Q4_K_M"

	if err := validateFinetuneConfig(config); err != nil {
		t.Fatalf("validateFinetuneConfig() error = %v", err)
	}
	if config.Output.Quantize != "q4_k_m" {
		t.Fatalf("normalized quantization = %q, want q4_k_m", config.Output.Quantize)
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
		Datasets: []config.Dataset{
			{Source: "test-dataset", Type: "alpaca"},
		},
		Config: config.FineTuneConfigSpec{
			Unsloth: config.FineTuneConfigUnslothSpec{
				MaxSeqLength:              2048,
				LoadIn4bit:                true,
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
			Quantize: "q4_k_m",
			Name:     "aikit-model",
		},
	}
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
