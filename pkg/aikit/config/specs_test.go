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

func TestNewFromBytesDefaultsFourBitLoading(t *testing.T) {
	tests := []struct {
		name           string
		config         string
		wantLoadIn4bit bool
	}{
		{name: "config omitted", wantLoadIn4bit: true},
		{name: "config null", config: "config:\n", wantLoadIn4bit: true},
		{name: "config empty", config: "config: {}\n", wantLoadIn4bit: true},
		{name: "unsloth null", config: "config:\n  unsloth:\n", wantLoadIn4bit: true},
		{name: "unsloth empty", config: "config:\n  unsloth: {}\n", wantLoadIn4bit: true},
		{name: "load setting false", config: "config:\n  unsloth:\n    loadIn4bit: false\n", wantLoadIn4bit: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := "apiVersion: v1alpha1\n" +
				"baseModel: test-model\n" +
				"datasets:\n" +
				"  - source: test-dataset\n" +
				"    type: alpaca\n" +
				tt.config
			_, fineTuneConfig, err := NewFromBytes([]byte(input))
			if err != nil {
				t.Fatalf("NewFromBytes() error = %v", err)
			}
			if fineTuneConfig == nil {
				t.Fatal("NewFromBytes() returned no fine-tune config")
			}
			if fineTuneConfig.Config.Unsloth.LoadIn4bit != tt.wantLoadIn4bit {
				t.Fatalf("loadIn4bit = %t, want %t", fineTuneConfig.Config.Unsloth.LoadIn4bit, tt.wantLoadIn4bit)
			}
		})
	}
}
