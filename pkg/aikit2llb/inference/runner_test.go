package inference

import (
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
)

func TestIsRunnerMode(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.InferenceConfig
		expected bool
	}{
		{
			name: "runner mode - backends with no models",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
			},
			expected: true,
		},
		{
			name: "not runner mode - backends with models",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
				Models: []config.Model{
					{Name: "test", Source: "http://example.com/model.gguf"},
				},
			},
			expected: false,
		},
		{
			name:     "not runner mode - no backends and no models",
			config:   &config.InferenceConfig{},
			expected: false,
		},
		{
			name: "not runner mode - no backends with models",
			config: &config.InferenceConfig{
				Models: []config.Model{
					{Name: "test", Source: "http://example.com/model.gguf"},
				},
			},
			expected: false,
		},
		{
			name: "runner mode - multiple backends with no models",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendDiffusers},
			},
			expected: true,
		},
		{
			name: "runner mode - vllm backend with no models",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendVLLM},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRunnerMode(tt.config)
			if result != tt.expected {
				t.Errorf("isRunnerMode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGenerateRunnerScript(t *testing.T) {
	tests := []struct {
		name           string
		config         *config.InferenceConfig
		expectContains []string
		expectMissing  []string
	}{
		{
			name: "llama-cpp backend script",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
			},
			expectContains: []string{
				`BACKEND="llama-cpp"`,
				".aikit-model-ref",
				"huggingface-cli",
				"curl -fL",
				"exec /usr/bin/local-ai",
			},
			expectMissing: []string{
				"--config-file",
				"--debug",
			},
		},
		{
			name: "diffusers backend script",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendDiffusers},
			},
			expectContains: []string{
				`BACKEND="diffusers"`,
				"aikit-model.yaml",
				"backend: diffusers",
				"exec /usr/bin/local-ai",
			},
		},
		{
			name: "vllm backend script",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendVLLM},
			},
			expectContains: []string{
				`BACKEND="vllm"`,
				"aikit-model.yaml",
				"backend: vllm",
				"exec /usr/bin/local-ai",
			},
		},
		{
			name: "script with debug enabled",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
				Debug:    true,
			},
			expectContains: []string{
				`BACKEND="llama-cpp"`,
				"--debug",
			},
		},
		{
			name: "script with config",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendLlamaCpp},
				Config:   "some-config",
			},
			expectContains: []string{
				"--config-file",
				"/config.yaml",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script := generateRunnerScript(tt.config)

			for _, expected := range tt.expectContains {
				if !strings.Contains(script, expected) {
					t.Errorf("generateRunnerScript() script does not contain %q\nScript:\n%s", expected, script)
				}
			}

			for _, missing := range tt.expectMissing {
				if strings.Contains(script, missing) {
					t.Errorf("generateRunnerScript() script should not contain %q\nScript:\n%s", missing, script)
				}
			}
		})
	}
}

func TestGenerateRunnerScriptArgParser(t *testing.T) {
	config := &config.InferenceConfig{
		Backends: []string{utils.BackendLlamaCpp},
	}

	script := generateRunnerScript(config)

	// The arg parser must handle --flag=value (single shift) differently
	// from --flag value (shift 2, consuming the next token as the value).
	// Without this, `docker run <image> --threads 4 model` would set MODEL=4.
	if !strings.Contains(script, `--*=*)`) {
		t.Error("arg parser should handle --flag=value style arguments with single shift")
	}

	// Should guard against trailing flags without values
	if !strings.Contains(script, `[[ $# -ge 2 ]]`) {
		t.Error("arg parser should guard against trailing flags without values")
	}

	// Should strip huggingface:// URI prefix for kubeairunway compatibility
	if !strings.Contains(script, `${MODEL#huggingface://}`) {
		t.Error("arg parser should strip huggingface:// URI prefix")
	}
}

func TestGenerateRunnerScriptModelConfig(t *testing.T) {
	config := &config.InferenceConfig{
		Backends: []string{utils.BackendLlamaCpp},
	}

	script := generateRunnerScript(config)

	// Should generate a model config YAML after downloading GGUF
	if !strings.Contains(script, "backend: llama-cpp") {
		t.Error("should generate a model config with llama-cpp backend")
	}
	if !strings.Contains(script, "parameters:") {
		t.Error("should include parameters section in generated config")
	}
	if !strings.Contains(script, ".yaml") {
		t.Error("should write a .yaml config file")
	}
}

func TestGenerateRunnerScriptUsageMessage(t *testing.T) {
	config := &config.InferenceConfig{
		Backends: []string{utils.BackendLlamaCpp},
	}

	script := generateRunnerScript(config)

	// Verify the usage message is present
	if !strings.Contains(script, "Usage: docker run") {
		t.Error("script should contain usage instructions")
	}
	if !strings.Contains(script, "HF_TOKEN") {
		t.Error("script should mention HF_TOKEN environment variable")
	}
}

func TestGenerateLlamaCppDownload(t *testing.T) {
	script := generateLlamaCppDownload()

	// Should use marker file for model-aware caching
	if !strings.Contains(script, ".aikit-model-ref") {
		t.Error("should use marker file for model-aware caching")
	}

	// Should detect model mismatch and re-download
	if !strings.Contains(script, "does not match requested model") {
		t.Error("should detect and handle model mismatch on cached volume")
	}

	// Should handle HTTP URLs
	if !strings.Contains(script, `"$MODEL" == http://*`) {
		t.Error("should handle HTTP URLs")
	}

	// Should handle HuggingFace repos
	if !strings.Contains(script, "huggingface-cli") {
		t.Error("should use huggingface-cli for HF repos")
	}

	// Should respect HF_TOKEN
	if !strings.Contains(script, "HF_TOKEN") {
		t.Error("should respect HF_TOKEN")
	}
}

func TestGenerateHFModelConfig(t *testing.T) {
	tests := []struct {
		name    string
		backend string
	}{
		{
			name:    "diffusers backend",
			backend: utils.BackendDiffusers,
		},
		{
			name:    "vllm backend",
			backend: utils.BackendVLLM,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script := generateHFModelConfig(tt.backend)

			// Should verify cached config matches the requested model
			if !strings.Contains(script, "grep -qF") {
				t.Error("should use fixed-string grep to verify cached config matches requested model")
			}

			// Should contain correct backend reference
			if !strings.Contains(script, "backend: "+tt.backend) {
				t.Errorf("should contain backend: %s", tt.backend)
			}

			// Should handle HF_TOKEN
			if !strings.Contains(script, "HF_TOKEN") {
				t.Error("should respect HF_TOKEN")
			}

			// Should handle model mismatch
			if !strings.Contains(script, "does not match requested model") {
				t.Error("should detect and handle model mismatch on cached volume")
			}
		})
	}
}
