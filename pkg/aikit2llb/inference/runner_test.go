package inference

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
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
	runnerCurlInvocation      = "curl -fL"
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
		{
			name: "runner mode - vllm-cpp backend with no models",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendVLLMCpp},
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
		{
			name:             "vllm-cpp installs downloader dependencies",
			backend:          utils.BackendVLLMCpp,
			wantDependencies: true,
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
				"http://azure.archive.ubuntu.com/ubuntu",
				"s|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g",
				"Acquire::Retries=5",
				"APT::Update::Error-Mode=any",
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
				runnerLlamaCppModelDir,
				runnerLlamaCppModelMarker,
				runnerHFCLIInvocation,
				runnerCurlInvocation,
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
				runnerConfigPath,
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
				runnerConfigPath,
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
			name: "vllm-cpp backend script",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendVLLMCpp},
			},
			expectContains: []string{
				`BACKEND="vllm-cpp"`,
				runnerVLLMCppModelDir,
				runnerVLLMCppModelMarker,
				runnerHFCLIInvocation,
				runnerCurlInvocation,
				"backend: vllm-cpp",
				"use_tokenizer_template: true",
				runnerLocalAIExec,
			},
			expectMissing: []string{
				runnerLegacyHFCLICommand,
				`model: ${MODEL}`,
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

			if !strings.Contains(script, `LOCAL_AI_ARGS=("--models-path" "$RUNNER_CONFIG_DIR")`) {
				t.Error("runner should scan only its owned config directory")
			}

			cmd := exec.Command("bash", "-n")
			cmd.Stdin = strings.NewReader(script)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("runner script has invalid shell syntax: %v: %s", err, output)
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
	tests := []struct {
		name          string
		backend       string
		wantExample   string
		rejectExample string
	}{
		{
			name:        utils.BackendLlamaCpp,
			backend:     utils.BackendLlamaCpp,
			wantExample: "--model org/model",
		},
		{
			name:          "vllm-cpp",
			backend:       utils.BackendVLLMCpp,
			wantExample:   "--model org/safetensors-model",
			rejectExample: "--model org/model\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script := generateRunnerScript(&config.InferenceConfig{Backends: []string{tt.backend}})
			cmd := exec.Command("bash")
			cmd.Stdin = strings.NewReader(script)
			output, err := cmd.CombinedOutput()
			if err == nil {
				t.Fatal("runner without a model unexpectedly succeeded")
			}

			usage := string(output)
			if !strings.Contains(usage, "Usage: docker run") {
				t.Error("script should contain usage instructions")
			}
			if !strings.Contains(usage, "HF_TOKEN") {
				t.Error("script should mention HF_TOKEN environment variable")
			}
			if !strings.Contains(usage, tt.wantExample) {
				t.Errorf("usage does not contain backend-compatible example %q: %s", tt.wantExample, usage)
			}
			if tt.rejectExample != "" && strings.Contains(usage, tt.rejectExample) {
				t.Errorf("usage contains incompatible example %q: %s", tt.rejectExample, usage)
			}
		})
	}
}

func TestGenerateLlamaCppDownload(t *testing.T) {
	script := generateLlamaCppDownload()

	// Should use marker file for model-aware caching
	if !strings.Contains(script, runnerLlamaCppModelMarker) {
		t.Error("should use marker file for model-aware caching")
	}
	if !strings.Contains(script, runnerLegacyModelMarker) {
		t.Error("should migrate the legacy model marker")
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
	if strings.Contains(script, "| head") {
		t.Error("should use find's native early exit instead of a pipe that can fail under pipefail")
	}
	if !strings.Contains(script, `find "$LLAMA_CPP_MODEL_DIR"`) || !strings.Contains(script, "-print -quit") {
		t.Error("should discover nested GGUF files only inside the llama-cpp-owned cache")
	}

	// Should respect HF_TOKEN
	if !strings.Contains(script, "HF_TOKEN") {
		t.Error("should respect HF_TOKEN")
	}
}

func TestLlamaCppRunnerMigratesLegacyModelCache(t *testing.T) {
	testRoot := t.TempDir()
	modelsDir := filepath.Join(testRoot, "models")
	binDir := filepath.Join(testRoot, "bin")
	for _, dir := range []string{modelsDir, binDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("create runner test directory %q: %v", dir, err)
		}
	}

	legacyMarker := filepath.Join(modelsDir, filepath.Base(runnerLegacyModelMarker))
	staleModel := filepath.Join(modelsDir, "stale.gguf")
	staleConfig := filepath.Join(modelsDir, "stale.yaml")
	userConfig := filepath.Join(modelsDir, "user.yaml")
	if err := os.WriteFile(legacyMarker, []byte("https://example.com/stale.gguf\n"), 0o600); err != nil {
		t.Fatalf("write legacy marker: %v", err)
	}
	if err := os.WriteFile(staleModel, []byte("stale"), 0o600); err != nil {
		t.Fatalf("write stale model: %v", err)
	}
	if err := os.WriteFile(staleConfig, []byte("name: stale\nbackend: llama-cpp\nparameters:\n  model: stale.gguf\n"), 0o600); err != nil {
		t.Fatalf("write stale generated config: %v", err)
	}
	if err := os.WriteFile(userConfig, []byte("name: user-owned\n"), 0o600); err != nil {
		t.Fatalf("write user config: %v", err)
	}

	writeRunnerStub(t, binDir, "curl", `#!/bin/bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$output" ]]
printf 'fresh' > "$output"
`)

	script := generateRunnerScript(&config.InferenceConfig{Backends: []string{utils.BackendLlamaCpp}})
	script = strings.ReplaceAll(script, "/models", modelsDir)
	script = strings.Replace(script, "exec /usr/bin/local-ai", "printf 'LOCAL_AI_ARG=%s\\n'", 1)
	model := "https://example.com/fresh.gguf"
	output, err := executeRunnerScript(t, script, binDir, model)
	if err != nil {
		t.Fatalf("run llama-cpp legacy cache migration: %v: %s", err, output)
	}

	if _, err := os.Stat(staleModel); err != nil {
		t.Fatalf("legacy GGUF without reliable ownership metadata should be preserved: %v", err)
	}
	if _, err := os.Stat(legacyMarker); !os.IsNotExist(err) {
		t.Fatalf("legacy marker remains after migration; stat error = %v", err)
	}
	for _, preserved := range []string{staleConfig, userConfig} {
		if _, err := os.Stat(preserved); err != nil {
			t.Fatalf("runner should preserve YAML outside its owned config directory %q: %v", preserved, err)
		}
	}
	freshModel := runnerPathInTestModels(modelsDir, runnerLlamaCppModelDir+"/fresh.gguf")
	assertRunnerFileContent(t, freshModel, "fresh")
	assertRunnerFileContent(t, filepath.Join(modelsDir, filepath.Base(runnerLlamaCppModelMarker)), model+"\n")
	configPath := runnerPathInTestModels(modelsDir, runnerConfigPath)
	wantConfig := "name: \"fresh\"\n" +
		"backend: llama-cpp\n" +
		"parameters:\n" +
		"  model: \"" + freshModel + "\"\n"
	assertRunnerFileContent(t, configPath, wantConfig)
	if !strings.Contains(string(output), "LOCAL_AI_ARG="+filepath.Dir(configPath)) {
		t.Fatalf("LocalAI does not scan the isolated runner config directory: %s", output)
	}
}

func TestLlamaCppRunnerReplacesOnlyOwnedCacheAndConfig(t *testing.T) {
	testRoot := t.TempDir()
	modelsDir := filepath.Join(testRoot, "models")
	binDir := filepath.Join(testRoot, "bin")
	llamaModelDir := runnerPathInTestModels(modelsDir, runnerLlamaCppModelDir)
	activeConfig := runnerPathInTestModels(modelsDir, runnerConfigPath)
	for _, dir := range []string{modelsDir, binDir, llamaModelDir, filepath.Dir(activeConfig)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("create runner test directory %q: %v", dir, err)
		}
	}

	oldOwnedModel := filepath.Join(llamaModelDir, "old.gguf")
	userModel := filepath.Join(modelsDir, "user.gguf")
	userConfig := filepath.Join(modelsDir, "user.yaml")
	seedFiles := map[string]string{
		oldOwnedModel: "old",
		userModel:     "user",
		userConfig:    "name: user-owned\n",
		activeConfig:  "name: old\nbackend: llama-cpp\nparameters:\n  model: old.gguf\n",
		filepath.Join(modelsDir, filepath.Base(runnerLlamaCppModelMarker)): "https://example.com/old.gguf\n",
	}
	for path, content := range seedFiles {
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatalf("seed runner file %q: %v", path, err)
		}
	}

	writeRunnerStub(t, binDir, "curl", `#!/bin/bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$output" ]]
printf 'fresh' > "$output"
`)

	script := generateRunnerScript(&config.InferenceConfig{Backends: []string{utils.BackendLlamaCpp}})
	script = strings.ReplaceAll(script, "/models", modelsDir)
	script = strings.Replace(script, "exec /usr/bin/local-ai", "exec true", 1)
	model := "https://example.com/fresh.gguf"
	output, err := executeRunnerScript(t, script, binDir, model)
	if err != nil {
		t.Fatalf("replace llama-cpp cache: %v: %s", err, output)
	}

	if _, err := os.Stat(oldOwnedModel); !os.IsNotExist(err) {
		t.Fatalf("old runner-owned GGUF remains after replacement; stat error = %v", err)
	}
	for _, preserved := range []string{userModel, userConfig} {
		if _, err := os.Stat(preserved); err != nil {
			t.Fatalf("runner should preserve unrelated file %q: %v", preserved, err)
		}
	}
	freshModel := filepath.Join(llamaModelDir, "fresh.gguf")
	assertRunnerFileContent(t, freshModel, "fresh")
	wantConfig := "name: \"fresh\"\n" +
		"backend: llama-cpp\n" +
		"parameters:\n" +
		"  model: \"" + freshModel + "\"\n"
	assertRunnerFileContent(t, activeConfig, wantConfig)
}

func TestLlamaCppRunnerDiscoversNestedGGUF(t *testing.T) {
	testRoot := t.TempDir()
	modelsDir := filepath.Join(testRoot, "models")
	binDir := filepath.Join(testRoot, "bin")
	for _, dir := range []string{modelsDir, binDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("create runner test directory %q: %v", dir, err)
		}
	}

	writeRunnerStub(t, binDir, "hf", `#!/bin/bash
set -euo pipefail
local_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--local-dir" ]]; then
    local_dir="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$local_dir" ]]
mkdir -p "$local_dir/quantized/q3"
printf 'GGUF' > "$local_dir/quantized/q3/nested.gguf"
`)

	script := generateRunnerScript(&config.InferenceConfig{Backends: []string{utils.BackendLlamaCpp}})
	script = strings.ReplaceAll(script, "/models", modelsDir)
	script = strings.Replace(script, "exec /usr/bin/local-ai", "exec true", 1)
	output, err := executeRunnerScript(t, script, binDir, "huggingface://org/nested-repo")
	if err != nil {
		t.Fatalf("run llama-cpp nested GGUF flow: %v: %s", err, output)
	}

	nestedModel := runnerPathInTestModels(modelsDir, runnerLlamaCppModelDir+"/quantized/q3/nested.gguf")
	wantConfig := "name: \"nested\"\n" +
		"backend: llama-cpp\n" +
		"parameters:\n" +
		"  model: \"" + nestedModel + "\"\n"
	assertRunnerFileContent(t, runnerPathInTestModels(modelsDir, runnerConfigPath), wantConfig)
	assertRunnerFileContent(t, filepath.Join(modelsDir, filepath.Base(runnerLlamaCppModelMarker)), "org/nested-repo\n")

	writeRunnerStub(t, binDir, "hf", "#!/bin/bash\nexit 91\n")
	output, err = executeRunnerScript(t, script, binDir, "huggingface://org/nested-repo")
	if err != nil {
		t.Fatalf("reuse cached nested llama-cpp GGUF: %v: %s", err, output)
	}
	if !strings.Contains(string(output), "Found cached model matching org/nested-repo") {
		t.Fatalf("nested GGUF cache hit was not detected: %s", output)
	}
}

func TestGenerateVLLMCppDownload(t *testing.T) {
	script := generateVLLMCppDownload()

	for _, expected := range []string{
		runnerVLLMCppModelMarker,
		`VLLM_CPP_MODEL_DIR="` + runnerVLLMCppModelDir + `"`,
		`hf download`,
		`--local-dir`,
		runnerCurlInvocation,
		`\.gguf$`,
		`backend: vllm-cpp`,
		`name: '${MODEL_NAME}'`,
		`model: '${LOCAL_MODEL_PATH}'`,
		`use_tokenizer_template: true`,
	} {
		if !strings.Contains(script, expected) {
			t.Errorf("vllm-cpp download script does not contain %q", expected)
		}
	}

	for _, unexpected := range []string{
		`model: ${MODEL}`,
		runnerLegacyHFCLICommand,
		`--include`,
	} {
		if strings.Contains(script, unexpected) {
			t.Errorf("vllm-cpp download script should not contain %q", unexpected)
		}
	}

	cmd := exec.Command("bash", "-n")
	cmd.Stdin = strings.NewReader(generateRunnerScript(&config.InferenceConfig{Backends: []string{utils.BackendVLLMCpp}}))
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("vllm-cpp runner script has invalid shell syntax: %v: %s", err, output)
	}
}

func TestRunnerBackendsUseSeparateModelMarkers(t *testing.T) {
	if runnerLlamaCppModelMarker == runnerVLLMCppModelMarker {
		t.Fatal("llama-cpp and vllm-cpp must not share a model cache marker")
	}

	llamaScript := generateLlamaCppDownload()
	vllmCppScript := generateVLLMCppDownload()
	if !strings.Contains(llamaScript, runnerLlamaCppModelMarker) || strings.Contains(llamaScript, runnerVLLMCppModelMarker) {
		t.Fatal("llama-cpp download script does not use only its backend-scoped marker")
	}
	if !strings.Contains(vllmCppScript, runnerVLLMCppModelMarker) || strings.Contains(vllmCppScript, runnerLlamaCppModelMarker) {
		t.Fatal("vllm-cpp download script does not use only its backend-scoped marker")
	}
	if !strings.Contains(llamaScript, `find "$LLAMA_CPP_MODEL_DIR"`) || strings.Contains(llamaScript, "| head") {
		t.Fatal("llama-cpp lookup must recursively search only its owned cache and avoid pipelines")
	}
}

func TestVLLMCppRunnerDownloadsHuggingFaceRepositoryLocally(t *testing.T) {
	script, modelsDir, binDir := prepareVLLMCppRunnerScript(t)
	hfArgsLog := filepath.Join(t.TempDir(), "hf-args")
	writeRunnerStub(t, binDir, "hf", `#!/bin/bash
set -euo pipefail
printf '%s\n' "$@" > "$HF_ARGS_LOG"
local_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--local-dir" ]]; then
    local_dir="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$local_dir" ]]
mkdir -p "$local_dir"
printf '{}\n' > "$local_dir/config.json"
printf 'weights\n' > "$local_dir/model.safetensors"
`)

	model := "huggingface://org/repo"
	output, err := executeRunnerScript(t, script, binDir, model, "HF_ARGS_LOG="+hfArgsLog, "HF_TOKEN=test-token")
	if err != nil {
		t.Fatalf("run vllm-cpp Hugging Face flow: %v: %s", err, output)
	}

	localModelDir := filepath.Join(modelsDir, strings.TrimPrefix(runnerVLLMCppModelDir, "/models/"))
	wantConfig := "name: 'repo'\n" +
		"backend: vllm-cpp\n" +
		"parameters:\n" +
		"  model: '" + localModelDir + "'\n" +
		"template:\n" +
		"  use_tokenizer_template: true\n"
	assertRunnerFileContent(t, runnerPathInTestModels(modelsDir, runnerConfigPath), wantConfig)
	assertRunnerFileContent(t, filepath.Join(modelsDir, filepath.Base(runnerVLLMCppModelMarker)), "org/repo\n")

	hfArgs, err := os.ReadFile(hfArgsLog)
	if err != nil {
		t.Fatalf("read hf arguments: %v", err)
	}
	for _, expected := range []string{"download\n", "org/repo\n", "--local-dir\n", localModelDir + "\n", "--exclude\n", "*.gguf\n", "--token\n", "test-token\n"} {
		if !strings.Contains(string(hfArgs), expected) {
			t.Errorf("hf arguments do not contain %q: %s", expected, hfArgs)
		}
	}

	if err := os.Remove(hfArgsLog); err != nil {
		t.Fatalf("remove first-run hf log: %v", err)
	}
	writeRunnerStub(t, binDir, "hf", "#!/bin/bash\nexit 91\n")
	output, err = executeRunnerScript(t, script, binDir, model)
	if err != nil {
		t.Fatalf("reuse cached vllm-cpp Hugging Face model: %v: %s", err, output)
	}
	if !strings.Contains(string(output), "Found cached vllm-cpp model matching org/repo") {
		t.Errorf("cache-hit output missing expected message: %s", output)
	}
	if _, err := os.Stat(hfArgsLog); !os.IsNotExist(err) {
		t.Errorf("hf downloader ran on cache hit; stat error = %v", err)
	}
}

func TestVLLMCppRunnerDownloadsDirectGGUFLocally(t *testing.T) {
	script, modelsDir, binDir := prepareVLLMCppRunnerScript(t)
	writeRunnerStub(t, binDir, "curl", `#!/bin/bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--output" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$output" ]]
printf 'GGUF' > "$output"
`)

	model := "https://example.com/null.gguf?download=1#weights"
	output, err := executeRunnerScript(t, script, binDir, model)
	if err != nil {
		t.Fatalf("run vllm-cpp GGUF flow: %v: %s", err, output)
	}

	localModelPath := filepath.Join(modelsDir, strings.TrimPrefix(runnerVLLMCppModelDir, "/models/"), "null.gguf")
	wantConfig := "name: 'null'\n" +
		"backend: vllm-cpp\n" +
		"parameters:\n" +
		"  model: '" + localModelPath + "'\n" +
		"template:\n" +
		"  use_tokenizer_template: true\n"
	assertRunnerFileContent(t, runnerPathInTestModels(modelsDir, runnerConfigPath), wantConfig)
	assertRunnerFileContent(t, localModelPath, "GGUF")
}

func TestVLLMCppRunnerIsolatesConfigFromStaleLlamaArtifacts(t *testing.T) {
	script, modelsDir, binDir := prepareVLLMCppRunnerScript(t)
	script = strings.Replace(script, "exec true", "printf 'LOCAL_AI_ARG=%s\\n'", 1)

	staleModel := filepath.Join(modelsDir, "stale.gguf")
	staleConfig := filepath.Join(modelsDir, "stale.yaml")
	activeConfig := runnerPathInTestModels(modelsDir, runnerConfigPath)
	if err := os.MkdirAll(filepath.Dir(activeConfig), 0o755); err != nil {
		t.Fatalf("create active config directory: %v", err)
	}
	if err := os.WriteFile(staleModel, []byte("stale"), 0o600); err != nil {
		t.Fatalf("write stale llama model: %v", err)
	}
	if err := os.WriteFile(staleConfig, []byte("name: stale\nbackend: llama-cpp\nparameters:\n  model: stale.gguf\n"), 0o600); err != nil {
		t.Fatalf("write stale llama config: %v", err)
	}
	if err := os.WriteFile(activeConfig, []byte("name: stale\nbackend: llama-cpp\n"), 0o600); err != nil {
		t.Fatalf("write previous active config: %v", err)
	}

	writeRunnerStub(t, binDir, "curl", `#!/bin/bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--output" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$output" ]]
printf 'GGUF' > "$output"
`)

	output, err := executeRunnerScript(t, script, binDir, "https://example.com/fresh.gguf")
	if err != nil {
		t.Fatalf("run vllm-cpp with stale llama artifacts: %v: %s", err, output)
	}

	for _, preserved := range []string{staleModel, staleConfig} {
		if _, err := os.Stat(preserved); err != nil {
			t.Fatalf("vllm-cpp should preserve unrelated llama artifact %q: %v", preserved, err)
		}
	}
	if !strings.Contains(string(output), "LOCAL_AI_ARG="+filepath.Dir(activeConfig)) {
		t.Fatalf("LocalAI does not scan only the isolated active config directory: %s", output)
	}
	wantConfig := "name: 'fresh'\n" +
		"backend: vllm-cpp\n" +
		"parameters:\n" +
		"  model: '" + runnerPathInTestModels(modelsDir, runnerVLLMCppModelDir+"/fresh.gguf") + "'\n" +
		"template:\n" +
		"  use_tokenizer_template: true\n"
	assertRunnerFileContent(t, activeConfig, wantConfig)
}

func TestVLLMCppRunnerRejectsRepositoryWithoutSafetensors(t *testing.T) {
	script, _, binDir := prepareVLLMCppRunnerScript(t)
	writeRunnerStub(t, binDir, "hf", `#!/bin/bash
set -euo pipefail
local_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--local-dir" ]]; then
    local_dir="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$local_dir" ]]
mkdir -p "$local_dir"
printf '{}\n' > "$local_dir/config.json"
printf 'GGUF' > "$local_dir/model.gguf"
`)

	output, err := executeRunnerScript(t, script, binDir, "huggingface://org/gguf-repo")
	if err == nil {
		t.Fatalf("GGUF-only Hugging Face repository unexpectedly succeeded: %s", output)
	}
	if !strings.Contains(string(output), "must contain config.json and safetensors weights") {
		t.Fatalf("unexpected GGUF-only repository error: %s", output)
	}
}

func TestVLLMCppRunnerRejectsUnsupportedModelReferences(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantError string
	}{
		{
			name:      "non-GGUF direct URL",
			model:     "https://example.com/model.safetensors",
			wantError: "direct URLs must point to a .gguf file",
		},
		{
			name:      "unsafe GGUF filename",
			model:     "https://example.com/model name.gguf",
			wantError: "direct URLs must point to a .gguf file",
		},
		{
			name:      "unsupported URL scheme",
			model:     "s3://bucket/model.gguf",
			wantError: "supports only huggingface:// repository references or HTTP(S) .gguf URLs",
		},
		{
			name:      "unsafe repository ID",
			model:     "org/repo/extra",
			wantError: "invalid Hugging Face repository ID",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script, _, binDir := prepareVLLMCppRunnerScript(t)
			output, err := executeRunnerScript(t, script, binDir, tt.model)
			if err == nil {
				t.Fatalf("model reference %q unexpectedly succeeded: %s", tt.model, output)
			}
			if !strings.Contains(string(output), tt.wantError) {
				t.Errorf("error output does not contain %q: %s", tt.wantError, output)
			}
		})
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
			if !strings.Contains(script, `grep -qxF "name: ${MODEL_NAME}" "$RUNNER_CONFIG"`) ||
				!strings.Contains(script, `grep -qxF "backend: `+tt.backend+`" "$RUNNER_CONFIG"`) ||
				!strings.Contains(script, `grep -qxF "  model: ${MODEL}" "$RUNNER_CONFIG"`) {
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

func prepareVLLMCppRunnerScript(t *testing.T) (string, string, string) {
	t.Helper()

	testRoot := t.TempDir()
	modelsDir := filepath.Join(testRoot, "models")
	binDir := filepath.Join(testRoot, "bin")
	for _, dir := range []string{modelsDir, binDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("create runner test directory %q: %v", dir, err)
		}
	}

	script := generateRunnerScript(&config.InferenceConfig{Backends: []string{utils.BackendVLLMCpp}})
	script = strings.ReplaceAll(script, "/models", modelsDir)
	script = strings.Replace(script, "exec /usr/bin/local-ai", "exec true", 1)

	return script, modelsDir, binDir
}

func runnerPathInTestModels(modelsDir, runnerPath string) string {
	return filepath.Join(modelsDir, strings.TrimPrefix(runnerPath, "/models/"))
}

func executeRunnerScript(t *testing.T, script, binDir, model string, extraEnv ...string) ([]byte, error) {
	t.Helper()

	cmd := exec.Command("bash", "-c", script, "aikit-runner", model)
	cmd.Env = append([]string{"PATH=" + binDir + string(os.PathListSeparator) + os.Getenv("PATH")}, extraEnv...)

	return cmd.CombinedOutput()
}

func writeRunnerStub(t *testing.T, binDir, name, content string) {
	t.Helper()

	//nolint:gosec // Runner command stubs must be executable by the test shell.
	if err := os.WriteFile(filepath.Join(binDir, name), []byte(content), 0o755); err != nil {
		t.Fatalf("write %s runner stub: %v", name, err)
	}
}

func assertRunnerFileContent(t *testing.T, path, want string) {
	t.Helper()

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read runner file %q: %v", path, err)
	}
	if got := string(content); got != want {
		t.Fatalf("runner file %q = %q, want %q", path, got, want)
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
