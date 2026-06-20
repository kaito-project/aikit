package config

import (
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/utils"
)

func TestInferenceConfigValidateChecksum(t *testing.T) {
	validSHA := strings.Repeat("a", 64)
	tests := []struct {
		name    string
		sha     string
		wantErr bool
	}{
		{name: "empty checksum is allowed", sha: "", wantErr: false},
		{name: "valid 64-char hex", sha: validSHA, wantErr: false},
		{name: "too short", sha: "abc123", wantErr: true},
		{name: "uppercase rejected", sha: strings.Repeat("A", 64), wantErr: true},
		{name: "algo-prefixed rejected", sha: "sha256:" + validSHA, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &InferenceConfig{
				APIVersion: utils.APIv1alpha1,
				Backends:   []string{utils.BackendLlamaCpp},
				Models:     []Model{{Name: "m", Source: "http://x/m.gguf", SHA256: tt.sha}},
			}
			err := c.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestInferenceConfigValidateAggregatesMembershipErrors(t *testing.T) {
	// Both an unknown backend and an unknown runtime should be reported together.
	c := &InferenceConfig{
		APIVersion: utils.APIv1alpha1,
		Runtime:    "bogus-runtime",
		Backends:   []string{"bogus-backend"},
	}
	err := c.Validate()
	if err == nil {
		t.Fatal("expected error for invalid backend and runtime")
	}
	msg := err.Error()
	if !strings.Contains(msg, "bogus-backend") || !strings.Contains(msg, "bogus-runtime") {
		t.Errorf("expected aggregated error to mention both backend and runtime, got: %s", msg)
	}
}
