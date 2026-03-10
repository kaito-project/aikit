package inference

import (
	"fmt"
	"slices"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

// isRunnerMode returns true when the config defines backends but no models,
// indicating a "runner" image that downloads models at container startup.
func isRunnerMode(c *config.InferenceConfig) bool {
	return len(c.Backends) > 0 && len(c.Models) == 0
}

// installRunnerDependencies installs packages needed for runtime model downloading
// (curl for HTTP, huggingface-cli for HuggingFace Hub).
func installRunnerDependencies(_ *config.InferenceConfig, s llb.State, merge llb.State, platform specs.Platform) (llb.State, llb.State) {
	savedState := s

	// Install curl for HTTP/HTTPS downloads and python3 + pip for huggingface-cli.
	// Some backends (diffusers/vllm) already install python, but llama-cpp does not,
	// so we always install the minimal set here.
	s = s.Run(
		utils.Sh("apt-get update && apt-get install --no-install-recommends -y curl python3 python3-pip && pip install --break-system-packages huggingface-hub[cli] 2>/dev/null || pip install huggingface-hub[cli] && apt-get clean"),
		llb.WithCustomNamef("Installing runner dependencies for platform %s/%s", platform.OS, platform.Architecture),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

// installRunnerEntrypoint writes the entrypoint script and creates the /models/
// directory with correct ownership for non-root compatibility.
func installRunnerEntrypoint(c *config.InferenceConfig, s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s

	script := generateRunnerScript(c)

	// Write the entrypoint script
	s = s.File(
		llb.Mkfile("/usr/local/bin/aikit-runner", 0o755, []byte(script)),
		llb.WithCustomName("Creating runner entrypoint script"),
	)

	// Create /models/ with UID 1000 ownership for non-root compatibility
	s = s.Run(
		utils.Sh("mkdir -p /models && chown 1000:1000 /models"),
		llb.WithCustomName("Creating /models directory with correct ownership"),
	).Root()

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

// generateRunnerScript produces the bash entrypoint script that downloads a model
// at container startup and then exec's into local-ai.
func generateRunnerScript(c *config.InferenceConfig) string {
	backend := utils.BackendLlamaCpp
	if len(c.Backends) > 0 {
		backend = c.Backends[0]
	}

	var sb strings.Builder
	sb.WriteString(`#!/bin/bash
set -euo pipefail

BACKEND="` + backend + `"

# Parse arguments: accept model as positional arg or --model flag
MODEL=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$MODEL" ]]; then
        MODEL="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: docker run <image> <model-ref>"
  echo ""
  echo "Examples:"
  echo "  docker run -p 8080:8080 <image> unsloth/gemma-3-1b-it-GGUF"
  echo "  docker run -p 8080:8080 <image> https://example.com/model.gguf"
  echo "  docker run -p 8080:8080 <image> --model unsloth/gemma-3-1b-it-GGUF"
  echo ""
  echo "Environment variables:"
  echo "  HF_TOKEN    - HuggingFace token for gated models"
  exit 1
fi

echo "AIKit Runner: backend=$BACKEND model=$MODEL"

`)

	// Backend-specific download logic
	if backend == utils.BackendLlamaCpp {
		sb.WriteString(generateLlamaCppDownload())
	} else if slices.Contains([]string{utils.BackendDiffusers, utils.BackendVLLM}, backend) {
		sb.WriteString(generateHFModelConfig(backend))
	}

	// Start LocalAI
	sb.WriteString(`
# Start local-ai
LOCAL_AI_ARGS=("--models-path" "/models")
`)

	// If config was baked in at build time, use it
	if c.Config != "" {
		sb.WriteString(`LOCAL_AI_ARGS+=("--config-file" "/config.yaml")
`)
	}

	if c.Debug {
		sb.WriteString(`LOCAL_AI_ARGS+=("--debug")
`)
	}

	sb.WriteString(`LOCAL_AI_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "Starting local-ai with args: ${LOCAL_AI_ARGS[*]}"
exec /usr/bin/local-ai "${LOCAL_AI_ARGS[@]}"
`)

	return sb.String()
}

// generateLlamaCppDownload generates the download logic for llama-cpp backend.
// It handles HuggingFace repos (downloading GGUF files) and direct HTTP URLs.
func generateLlamaCppDownload() string {
	return `# Check if models already exist (volume mount caching)
EXISTING_MODELS=$(find /models -name "*.gguf" -type f 2>/dev/null | head -1)
if [[ -n "$EXISTING_MODELS" ]]; then
  echo "Found existing model files in /models, skipping download"
else
  if [[ "$MODEL" == http://* ]] || [[ "$MODEL" == https://* ]]; then
    # Direct HTTP/HTTPS download
    echo "Downloading model from URL: $MODEL"
    FILENAME=$(basename "$MODEL")
    curl -L --progress-bar -o "/models/$FILENAME" "$MODEL"
  else
    # HuggingFace repo - download GGUF files
    echo "Downloading GGUF files from HuggingFace: $MODEL"
    HF_ARGS=("download" "$MODEL" "--local-dir" "/models" "--include" "*.gguf")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_ARGS+=("--token" "$HF_TOKEN")
    fi
    huggingface-cli "${HF_ARGS[@]}"
  fi
  echo "Download complete"
fi
`
}

// generateHFModelConfig generates the download and config logic for diffusers/vllm backends.
// These backends pass the HuggingFace model ID through to LocalAI config at runtime.
func generateHFModelConfig(backend string) string {
	return fmt.Sprintf(`# Check if model config already exists (volume mount caching)
if [[ -f "/models/aikit-model.yaml" ]]; then
  echo "Found existing model config in /models, skipping setup"
else
  # For %[1]s backend, generate a LocalAI model config pointing to the HF model
  MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
  echo "Generating LocalAI config for %[1]s backend with model: $MODEL"
  cat > /models/aikit-model.yaml <<MODELEOF
name: ${MODEL_NAME}
backend: %[1]s
parameters:
  model: ${MODEL}
MODELEOF
  echo "Config generated at /models/aikit-model.yaml"

  # Pre-download if using HuggingFace reference (not a URL)
  if [[ "$MODEL" != http://* ]] && [[ "$MODEL" != https://* ]]; then
    echo "Pre-downloading model from HuggingFace: $MODEL"
    HF_ARGS=("download" "$MODEL" "--local-dir" "/models/${MODEL_NAME}")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_ARGS+=("--token" "$HF_TOKEN")
    fi
    huggingface-cli "${HF_ARGS[@]}" || echo "Pre-download skipped (model will be downloaded by backend)"
  fi
fi
`, backend)
}
