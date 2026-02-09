package inference

import (
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
)

func TestGenerateLocalAIConfig(t *testing.T) {
	tests := []struct {
		name           string
		config         *config.InferenceConfig
		existingConfig string
		expected       string
	}{
		{
			name: "preload disabled",
			config: &config.InferenceConfig{
				PreloadModels: false,
				Models: []config.Model{
					{Name: "test-model", Source: "test.gguf"},
				},
			},
			existingConfig: "existing config",
			expected:       "existing config",
		},
		{
			name: "preload enabled with URL source",
			config: &config.InferenceConfig{
				PreloadModels: true,
				Models: []config.Model{
					{Name: "llama-model", Source: "https://example.com/model.gguf"},
				},
			},
			existingConfig: "",
			expected: `preload_models:
  - id: llama-model
    name: model.gguf
    preload: true
`,
		},
		{
			name: "preload enabled with local source",
			config: &config.InferenceConfig{
				PreloadModels: true,
				Models: []config.Model{
					{Name: "local-model", Source: "/path/to/model.gguf"},
				},
			},
			existingConfig: "",
			expected: `preload_models:
  - id: local-model
    name: model.gguf
    preload: true
`,
		},
		{
			name: "preload enabled with existing config",
			config: &config.InferenceConfig{
				PreloadModels: true,
				Models: []config.Model{
					{Name: "test-model", Source: "model.gguf"},
				},
			},
			existingConfig: "existing: config",
			expected: `existing: config
preload_models:
  - id: test-model
    name: model.gguf
    preload: true
`,
		},
		{
			name: "multiple models preload",
			config: &config.InferenceConfig{
				PreloadModels: true,
				Models: []config.Model{
					{Name: "model1", Source: "model1.gguf"},
					{Name: "model2", Source: "https://example.com/model2.gguf"},
				},
			},
			existingConfig: "",
			expected: `preload_models:
  - id: model1
    name: model1.gguf
    preload: true
  - id: model2
    name: model2.gguf
    preload: true
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateLocalAIConfig(tt.config, tt.existingConfig)
			if result != tt.expected {
				t.Errorf("generateLocalAIConfig() = %q, expected %q", result, tt.expected)
			}
		})
	}
}
