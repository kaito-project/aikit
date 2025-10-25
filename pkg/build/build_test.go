package build

import (
	"reflect"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
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
				Backends:   []string{"exllama2"},
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
			name: "valid exllama2 backend with cpu runtime",
			args: args{c: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"exllama2"},
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
			name: "invalid backend combination",
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
			name: "exllama2 backend with arm64 platform - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"exllama2"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "arm64", OS: "linux"},
			},
			wantErr: true,
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
			name: "exllama2 backend with amd64 platform - should pass",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"exllama2"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
			},
			wantErr: false,
		},
		{
			name: "mixed platforms with exllama2 backend - should fail",
			config: &config.InferenceConfig{
				APIVersion: "v1alpha1",
				Backends:   []string{"exllama2"},
			},
			targetPlatforms: []*specs.Platform{
				{Architecture: "amd64", OS: "linux"},
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
	type args struct {
		c *config.FineTuneConfig
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name:    "no config",
			args:    args{c: &config.FineTuneConfig{}},
			wantErr: true,
		},
		{
			name: "unsupported api version",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v10",
			}},
			wantErr: true,
		},
		{
			name: "invalid target",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v1alpha1",
				Target:     "foo",
			}},
			wantErr: true,
		},
		{
			name: "no datasets",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v1alpha1",
				Target:     "unsloth",
			}},
			wantErr: true,
		},
		{
			name: "invalid dataset type",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v1alpha1",
				Target:     "unsloth",
				Datasets: []config.Dataset{
					{
						Source: "foo",
						Type:   "bar",
					},
				},
			}},
			wantErr: true,
		},
		{
			name: "valid dataset type",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v1alpha1",
				Target:     "unsloth",
				Datasets: []config.Dataset{
					{
						Source: "foo",
						Type:   "alpaca",
					},
				},
			}},
			wantErr: false,
		},
		{
			name: "multiple datasets",
			args: args{c: &config.FineTuneConfig{
				APIVersion: "v1alpha1",
				Target:     "unsloth",
				Datasets: []config.Dataset{
					{
						Source: "foo",
						Type:   "alpaca",
					},
					{
						Source: "bar",
						Type:   "alpaca",
					},
				},
			}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateFinetuneConfig(tt.args.c); (err != nil) != tt.wantErr {
				t.Errorf("validateFineTuneConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_defaultsUnslothConfig(t *testing.T) {
	type args struct {
		c *config.FineTuneConfig
	}
	tests := []struct {
		name string
		args args
		want *config.FineTuneConfig
	}{
		{
			name: "no config",
			args: args{c: &config.FineTuneConfig{}},
			want: &config.FineTuneConfig{
				Config: config.FineTuneConfigSpec{
					Unsloth: config.FineTuneConfigUnslothSpec{
						Packing:                   false,
						MaxSeqLength:              2048,
						LoadIn4bit:                false,
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
			},
		},
		{
			name: "with config",
			args: args{c: &config.FineTuneConfig{
				Config: config.FineTuneConfigSpec{
					Unsloth: config.FineTuneConfigUnslothSpec{
						Packing:                   true,
						MaxSeqLength:              1024,
						LoadIn4bit:                true,
						BatchSize:                 4,
						GradientAccumulationSteps: 8,
						WarmupSteps:               20,
						MaxSteps:                  120,
						LearningRate:              0.0004,
						LoggingSteps:              2,
						Optimizer:                 "adamw_16bit",
						WeightDecay:               0.02,
						LrSchedulerType:           "cosine",
						Seed:                      24,
					},
				},
			}},
			want: &config.FineTuneConfig{
				Config: config.FineTuneConfigSpec{
					Unsloth: config.FineTuneConfigUnslothSpec{
						Packing:                   true,
						MaxSeqLength:              1024,
						LoadIn4bit:                true,
						BatchSize:                 4,
						GradientAccumulationSteps: 8,
						WarmupSteps:               20,
						MaxSteps:                  120,
						LearningRate:              0.0004,
						LoggingSteps:              2,
						Optimizer:                 "adamw_16bit",
						WeightDecay:               0.02,
						LrSchedulerType:           "cosine",
						Seed:                      24,
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaultsUnslothConfig(tt.args.c)
			if !reflect.DeepEqual(tt.args.c, tt.want) {
				t.Errorf("defaultsUnslothConfig() = %v, want %v", tt.args.c, tt.want)
			}
		})
	}
}
