package config

import (
	"reflect"
	"testing"

	"github.com/sozercan/aikit/pkg/utils"
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
runtime: avx512
backends:
- exllama
- stablediffusion
models:
- name: test
  source: foo
`)},
			want: &InferenceConfig{
				APIVersion: utils.APIv1alpha1,
				Runtime:    utils.RuntimeCPUAVX512,
				Backends: []string{
					utils.BackendExllama,
					utils.BackendStableDiffusion,
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
