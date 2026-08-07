package config

import (
	"reflect"
	"testing"

	"github.com/kaito-project/aikit/pkg/utils"
)

func TestNewFromBytes(t *testing.T) {
	type args struct {
		b []byte
	}
	tests := []struct {
		name    string
		args    args
		want    *InferenceConfig
		wantErr bool
	}{
		{
			name: "valid yaml",
			args: args{b: []byte(`
apiVersion: v1alpha1
runtime: cuda
backends:
- diffusers
models:
- name: test
  source: foo
`)},
			want: &InferenceConfig{
				APIVersion: utils.APIv1alpha1,
				Runtime:    utils.RuntimeNVIDIA,
				Backends: []string{
					utils.BackendDiffusers,
				},
				Models: []Model{
					{
						Name:   "test",
						Source: "foo",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid yaml",
			args: args{b: []byte(`
foo
`)},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			infCfg, _, err := NewFromBytes(tt.args.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewFromBytes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(infCfg, tt.want) {
				t.Errorf("NewFromBytes() = %v, want %v", infCfg, tt.want)
			}
		})
	}
}

func TestNewFromBytesNormalizesFineTuneDefaults(t *testing.T) {
	tests := []struct {
		name     string
		sections string
	}{
		{name: "sections omitted"},
		{name: "sections null", sections: "config:\noutput:\n"},
		{name: "sections empty", sections: "config: {}\noutput: {}\n"},
		{name: "unsloth null", sections: "config:\n  unsloth:\noutput: {}\n"},
		{name: "unsloth empty", sections: "config:\n  unsloth: {}\noutput: {}\n"},
	}

	wantUnsloth := FineTuneConfigUnslothSpec{
		Packing:                   false,
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
	}
	wantOutput := FineTuneOutputSpec{Quantize: "q4_k_m", Name: "aikit-model"}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := "apiVersion: v1alpha1\n" +
				"baseModel: test-model\n" +
				"datasets:\n" +
				"  - source: test-dataset\n" +
				"    type: alpaca\n" +
				tt.sections
			_, fineTuneConfig, err := NewFromBytes([]byte(input))
			if err != nil {
				t.Fatalf("NewFromBytes() error = %v", err)
			}
			if fineTuneConfig == nil {
				t.Fatal("NewFromBytes() returned no fine-tune config")
			}
			if !reflect.DeepEqual(fineTuneConfig.Config.Unsloth, wantUnsloth) {
				t.Errorf("unsloth config = %#v, want %#v", fineTuneConfig.Config.Unsloth, wantUnsloth)
			}
			if !reflect.DeepEqual(fineTuneConfig.Output, wantOutput) {
				t.Errorf("output config = %#v, want %#v", fineTuneConfig.Output, wantOutput)
			}
		})
	}
}

func TestNewFromBytesPreservesExplicitFineTuneZeroValues(t *testing.T) {
	input := []byte(`
apiVersion: v1alpha1
baseModel: test-model
datasets:
  - source: test-dataset
    type: alpaca
config:
  unsloth:
    packing: false
    maxSeqLength: 0
    loadIn4bit: false
    loss: ""
    batchSize: 0
    gradientAccumulationSteps: 0
    warmupSteps: 0
    maxSteps: 0
    learningRate: 0
    loggingSteps: 0
    optimizer: ""
    weightDecay: 0
    lrSchedulerType: ""
    seed: 0
output:
  quantize: ""
  name: ""
`)

	_, fineTuneConfig, err := NewFromBytes(input)
	if err != nil {
		t.Fatalf("NewFromBytes() error = %v", err)
	}
	if fineTuneConfig == nil {
		t.Fatal("NewFromBytes() returned no fine-tune config")
	}

	wantUnsloth := FineTuneConfigUnslothSpec{}
	if !reflect.DeepEqual(fineTuneConfig.Config.Unsloth, wantUnsloth) {
		t.Errorf("unsloth config = %#v, want explicit zero values %#v", fineTuneConfig.Config.Unsloth, wantUnsloth)
	}
	if fineTuneConfig.Output != (FineTuneOutputSpec{}) {
		t.Errorf("output config = %#v, want explicit empty values", fineTuneConfig.Output)
	}
}

func TestNewFromBytesPreservesExplicitFineTuneLoss(t *testing.T) {
	input := []byte(`
apiVersion: v1alpha1
baseModel: test-model
datasets:
  - source: test-dataset
    type: messages
config:
  unsloth:
    loss: response
`)

	_, fineTuneConfig, err := NewFromBytes(input)
	if err != nil {
		t.Fatalf("NewFromBytes() error = %v", err)
	}
	if fineTuneConfig == nil {
		t.Fatal("NewFromBytes() returned no fine-tune config")
	}
	if got := fineTuneConfig.Config.Unsloth.Loss; got != utils.SFTLossResponse {
		t.Fatalf("unsloth loss = %q, want %q", got, utils.SFTLossResponse)
	}
}

func TestNewFromBytesPreservesExplicitEmptyFineTuneLoss(t *testing.T) {
	input := []byte(`
apiVersion: v1alpha1
baseModel: test-model
datasets:
  - source: test-dataset
    type: messages
config:
  unsloth:
    loss: ""
`)

	_, fineTuneConfig, err := NewFromBytes(input)
	if err != nil {
		t.Fatalf("NewFromBytes() error = %v", err)
	}
	if fineTuneConfig == nil {
		t.Fatal("NewFromBytes() returned no fine-tune config")
	}
	if got := fineTuneConfig.Config.Unsloth.Loss; got != "" {
		t.Fatalf("unsloth loss = %q, want explicit empty value", got)
	}
}

func TestNewFromBytesDefaultsNullFineTuneLoss(t *testing.T) {
	for _, loss := range []string{"loss: null", "loss:"} {
		t.Run(loss, func(t *testing.T) {
			input := []byte(`
apiVersion: v1alpha1
baseModel: test-model
datasets:
  - source: test-dataset
    type: messages
config:
  unsloth:
    ` + loss + "\n")

			_, fineTuneConfig, err := NewFromBytes(input)
			if err != nil {
				t.Fatalf("NewFromBytes() error = %v", err)
			}
			if fineTuneConfig == nil {
				t.Fatal("NewFromBytes() returned no fine-tune config")
			}
			if got := fineTuneConfig.Config.Unsloth.Loss; got != utils.SFTLossAll {
				t.Fatalf("unsloth loss = %q, want null default %q", got, utils.SFTLossAll)
			}
		})
	}
}
