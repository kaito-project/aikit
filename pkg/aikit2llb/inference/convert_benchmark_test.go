package inference

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func BenchmarkAikit2LLBMarshal(b *testing.B) {
	benchmarks := []struct {
		name          string
		templateCount int
	}{
		{name: "LocalModel", templateCount: 0},
		{name: "LlamaFixture", templateCount: 4},
		{name: "ManyPromptTemplates", templateCount: 100},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			cfg := benchmarkInferenceConfig(benchmark.templateCount)
			platform := &specs.Platform{OS: "linux", Architecture: "arm64"}
			ctx := context.Background()

			state, _, err := Aikit2LLB(cfg, platform)
			if err != nil {
				b.Fatal(err)
			}
			definition, err := state.Marshal(ctx)
			if err != nil {
				b.Fatal(err)
			}
			definitionBytes := 0
			for _, op := range definition.Def {
				definitionBytes += len(op)
			}
			b.ReportAllocs()
			b.ResetTimer()

			for b.Loop() {
				state, _, err = Aikit2LLB(cfg, platform)
				if err != nil {
					b.Fatal(err)
				}
				definition, err = state.Marshal(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(float64(len(definition.Def)), "ops/graph")
			b.ReportMetric(float64(definitionBytes), "opbytes/graph")
		})
	}
}

func benchmarkInferenceConfig(templateCount int) *config.InferenceConfig {
	templates := make([]config.PromptTemplate, templateCount)
	templateBody := strings.Repeat("prompt {{.Input}} ", 32)
	for i := range templates {
		templates[i] = config.PromptTemplate{
			Name:     fmt.Sprintf("template-%03d", i),
			Template: templateBody,
		}
	}

	return &config.InferenceConfig{
		APIVersion: "v1alpha1",
		Backends:   []string{utils.BackendLlamaCpp},
		Models: []config.Model{
			{
				Name:            "tiny",
				Source:          "tiny.gguf",
				PromptTemplates: templates,
			},
		},
		Config: "- name: tiny\n  backend: llama-cpp\n  parameters:\n    model: tiny.gguf\n",
	}
}
