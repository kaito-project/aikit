package inference

import (
	"context"
	"os/exec"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	runnerHFArgsAssignment    = "HF_ARGS="
	runnerHFCLIInvocation     = "hf download"
	runnerLegacyHFCLICommand  = "huggingface-cli"
	runnerLocalAIExec         = "exec /usr/bin/local-ai"
	runnerLocalDirFlag        = "--local-dir"
	runnerPredownloadMessage  = "Pre-downloading model"
	runnerTestInferenceModel  = "test"
	runnerTestInferenceSource = "http://example.com/model.gguf"
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
					{Name: runnerTestInferenceModel, Source: runnerTestInferenceSource},
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
					{Name: runnerTestInferenceModel, Source: runnerTestInferenceSource},
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

func TestInstallRunnerDependencies(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	tests := []struct {
		name             string
		backend          string
		wantDependencies bool
	}{
		{
			name:             "llama-cpp installs downloader dependencies",
			backend:          utils.BackendLlamaCpp,
			wantDependencies: true,
		},
		{
			name:    "diffusers uses bundled downloader",
			backend: utils.BackendDiffusers,
		},
		{
			name:    "vllm uses bundled downloader",
			backend: utils.BackendVLLM,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state, _ := installRunnerDependencies(
				&config.InferenceConfig{Backends: []string{tt.backend}},
				llb.Scratch(),
				llb.Scratch(),
				platform,
			)
			commands := runnerExecCommands(t, state)

			if !tt.wantDependencies {
				if len(commands) != 0 {
					t.Fatalf("runner dependency commands = %q, want none", commands)
				}
				return
			}

			if len(commands) != 1 {
				t.Fatalf("runner dependency commands = %q, want exactly one", commands)
			}
			command := commands[0]
			for _, expected := range []string{
				"apt-get update",
				"apt-get install --no-install-recommends -y curl ca-certificates python3 python3-pip",
				"huggingface-hub==" + runnerHuggingFaceHubVersion,
				"apt-get clean",
				"rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /root/.cache/pip",
			} {
				if !strings.Contains(command, expected) {
					t.Errorf("runner dependency command does not contain %q: %s", expected, command)
				}
			}
			if count := strings.Count(command, "--no-cache-dir"); count != 2 {
				t.Errorf("--no-cache-dir count = %d, want 2 in both pip install paths: %s", count, command)
			}
			if count := strings.Count(command, "--no-compile"); count != 2 {
				t.Errorf("--no-compile count = %d, want 2 in both pip install paths: %s", count, command)
			}
			if strings.Contains(command, "huggingface-hub[cli]") {
				t.Errorf("runner dependency command should install the pinned core package without the legacy CLI extra: %s", command)
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
				runnerHFCLIInvocation,
				"curl -fL",
				runnerLocalAIExec,
			},
			expectMissing: []string{
				"--config-file",
				"--debug",
				runnerLegacyHFCLICommand,
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
				runnerLocalAIExec,
			},
			expectMissing: []string{
				runnerHFCLIInvocation,
				runnerLegacyHFCLICommand,
				runnerPredownloadMessage,
				runnerHFArgsAssignment,
				runnerLocalDirFlag,
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
				runnerLocalAIExec,
			},
			expectMissing: []string{
				runnerHFCLIInvocation,
				runnerLegacyHFCLICommand,
				runnerPredownloadMessage,
				runnerHFArgsAssignment,
				runnerLocalDirFlag,
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
	if !strings.Contains(script, runnerHFCLIInvocation) {
		t.Error("should use hf download for HF repos")
	}
	if strings.Contains(script, runnerLegacyHFCLICommand) {
		t.Error("should not use the legacy huggingface-cli command")
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

			// Model names should use the normalized repository basename so API callers
			// do not need to include the HuggingFace organization prefix.
			if !strings.Contains(script, `MODEL_NAME_SOURCE="${MODEL%%\#*}"`) ||
				!strings.Contains(script, `MODEL_NAME_SOURCE="${MODEL_NAME_SOURCE%%\?*}"`) ||
				!strings.Contains(script, `MODEL_NAME_SOURCE="${MODEL_NAME_SOURCE%/}"`) {
				t.Error("should normalize fragments, queries, and trailing slashes before deriving the model name")
			}

			// Cached configs should match the alias, backend, and requested model source.
			if !strings.Contains(script, `grep -qxF "name: ${MODEL_NAME}"`) ||
				!strings.Contains(script, `grep -qxF "backend: `+tt.backend+`"`) ||
				!strings.Contains(script, `grep -qxF "  model: ${MODEL}"`) {
				t.Error("should validate the cached model name, backend, and source")
			}

			if !strings.Contains(script, "backend: "+tt.backend) {
				t.Errorf("should contain backend: %s", tt.backend)
			}

			// Cache logs should identify the backend that matched or changed.
			if !strings.Contains(script, "Found existing "+tt.backend+" model config") ||
				!strings.Contains(script, "does not match requested backend/model ("+tt.backend) {
				t.Error("should identify the backend in cache hit and mismatch logs")
			}

			// Bundled Python backends should download through their own Hugging Face cache.
			for _, unexpected := range []string{
				runnerHFCLIInvocation,
				runnerLegacyHFCLICommand,
				runnerPredownloadMessage,
				runnerHFArgsAssignment,
				runnerLocalDirFlag,
				"HF_TOKEN",
			} {
				if strings.Contains(script, unexpected) {
					t.Errorf("should not contain external downloader fragment %q", unexpected)
				}
			}
		})
	}
}

func TestRunnerModelNameScript(t *testing.T) {
	const wantRepositoryName = "repo"

	tests := []struct {
		name      string
		model     string
		want      string
		wantError bool
	}{
		{name: "huggingface repository", model: "org/repo", want: wantRepositoryName},
		{name: "trailing slash", model: "org/repo/", want: wantRepositoryName},
		{name: "URL query containing slashes", model: "https://example.com/models/repo/?revision=refs/pr/1", want: wantRepositoryName},
		{name: "URL fragment containing slashes", model: "https://example.com/models/repo#refs/pr/1", want: wantRepositoryName},
		{name: "empty normalized path", model: "////?revision=refs/pr/1", wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("bash", "-eu", "-o", "pipefail", "-c", runnerModelNameScript+"\nprintf '%s' \"$MODEL_NAME\"")
			cmd.Env = append(cmd.Environ(), "MODEL="+tt.model)
			output, err := cmd.CombinedOutput()
			if tt.wantError {
				if err == nil {
					t.Fatalf("model %q unexpectedly produced %q", tt.model, output)
				}
				return
			}
			if err != nil {
				t.Fatalf("normalize model %q: %v: %s", tt.model, err, output)
			}
			if got := string(output); got != tt.want {
				t.Fatalf("normalize model %q = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

func runnerExecCommands(t *testing.T, state llb.State) []string {
	t.Helper()

	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal runner state: %v", err)
	}

	commands := make([]string, 0)
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal runner LLB op: %v", err)
		}
		if exec := op.GetExec(); exec != nil {
			commands = append(commands, strings.Join(exec.Meta.Args, " "))
		}
	}
	return commands
}
