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
		{
			name: "unknown field is rejected (strict)",
			args: args{b: []byte(`
apiVersion: v1alpha1
bckends:
- llama-cpp
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

func TestNewFromBytes_Discriminator(t *testing.T) {
	// A finetune document without a "config:" block must be classified as
	// finetune (not silently misdetected as an empty inference config).
	finetuneNoConfigBlock := []byte(`
apiVersion: v1alpha1
baseModel: unsloth/Meta-Llama-3.1-8B
datasets:
  - source: "yahma/alpaca-cleaned"
    type: alpaca
`)
	inf, ft, err := NewFromBytes(finetuneNoConfigBlock)
	if err != nil {
		t.Fatalf("NewFromBytes() unexpected error = %v", err)
	}
	if inf != nil {
		t.Errorf("NewFromBytes() classified finetune doc as inference: %+v", inf)
	}
	if ft == nil {
		t.Fatalf("NewFromBytes() did not return a finetune config")
	}
	if ft.BaseModel != "unsloth/Meta-Llama-3.1-8B" {
		t.Errorf("NewFromBytes() baseModel = %q, want %q", ft.BaseModel, "unsloth/Meta-Llama-3.1-8B")
	}
}
