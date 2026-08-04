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

const (
	runnerHuggingFaceHubVersion = "1.26.0"
	runnerDependenciesCommand   = "sed -i 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g; " +
		"s|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && " +
		"apt-get -o Acquire::Retries=5 -o APT::Update::Error-Mode=any update && " +
		"apt-get install --no-install-recommends -y curl ca-certificates python3 python3-pip && " +
		"(pip install --no-cache-dir --no-compile --break-system-packages huggingface-hub==" + runnerHuggingFaceHubVersion + " 2>/dev/null || " +
		"pip install --no-cache-dir --no-compile huggingface-hub==" + runnerHuggingFaceHubVersion + ") && " +
		"apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /root/.cache/pip"
)

// isRunnerMode returns true when the config defines backends but no models,
// indicating a "runner" image that resolves models at runtime.
func isRunnerMode(c *config.InferenceConfig) bool {
	return len(c.Backends) > 0 && len(c.Models) == 0
}

// installRunnerDependencies installs packages needed for runtime model downloading.
func installRunnerDependencies(c *config.InferenceConfig, s llb.State, merge llb.State, platform specs.Platform) (llb.State, llb.State) {
	if runnerBackend(c) != utils.BackendLlamaCpp {
		return s, merge
	}

	savedState := s

	// Install curl for HTTP/HTTPS downloads and the hf CLI for repository downloads.
	// Note: Runner mode is not supported for Apple Silicon (validated in build).
	s = s.Run(
		utils.Sh(runnerDependenciesCommand),
		llb.WithCustomNamef("Installing runner dependencies for platform %s/%s", platform.OS, platform.Architecture),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

func runnerBackend(c *config.InferenceConfig) string {
	if len(c.Backends) > 0 {
		return c.Backends[0]
	}
	return utils.BackendLlamaCpp
}

// installRunnerEntrypoint writes the entrypoint script and creates the /models/
// directory with correct ownership for non-root compatibility.
func installRunnerEntrypoint(c *config.InferenceConfig, s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s

	script := generateRunnerScript(c)

	// Write the entrypoint script
	s = s.File(
		llb.Mkfile(runnerEntrypointPath, 0o755, []byte(script)),
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
	backend := runnerBackend(c)

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
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value"; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --*=*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --*)
      if [[ $# -ge 2 ]]; then
        EXTRA_ARGS+=("$1" "$2")
        shift 2
      else
        EXTRA_ARGS+=("$1")
        shift
      fi
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

# Strip URI scheme prefixes (e.g. huggingface://org/repo -> org/repo)
# kubeairunway passes model IDs with the huggingface:// prefix.
MODEL="${MODEL#huggingface://}"

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

	sb.WriteString(`if ((${#EXTRA_ARGS[@]})); then
  LOCAL_AI_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Starting local-ai with args: ${LOCAL_AI_ARGS[*]}"
exec /usr/bin/local-ai "${LOCAL_AI_ARGS[@]}"
`)

	return sb.String()
}

// generateLlamaCppDownload generates the download logic for llama-cpp backend.
// It handles HuggingFace repos (downloading GGUF files) and direct HTTP URLs.
func generateLlamaCppDownload() string {
	return `# Check if the requested model already exists (volume mount caching)
# Write a marker file so we can detect model mismatches on reuse.
MODEL_MARKER="/models/.aikit-model-ref"
if [[ -f "$MODEL_MARKER" ]] && [[ "$(cat "$MODEL_MARKER")" == "$MODEL" ]]; then
  echo "Found cached model matching $MODEL in /models, skipping download"
else
  # Different model requested or no marker — clean and re-download
  if [[ -f "$MODEL_MARKER" ]]; then
    echo "Cached model ($(cat "$MODEL_MARKER")) does not match requested model ($MODEL), re-downloading"
    rm -f /models/*.gguf "$MODEL_MARKER"
  fi
  if [[ "$MODEL" == http://* ]] || [[ "$MODEL" == https://* ]]; then
    # Direct HTTP/HTTPS download
    echo "Downloading model from URL: $MODEL"
    FILENAME=$(basename "$MODEL")
    curl -fL --progress-bar -o "/models/$FILENAME" "$MODEL"
  else
    # HuggingFace repo - download GGUF files
    echo "Downloading GGUF files from HuggingFace: $MODEL"
    HF_ARGS=("$MODEL" "--local-dir" "/models" "--include" "*.gguf")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_ARGS+=("--token" "$HF_TOKEN")
    fi
    hf download "${HF_ARGS[@]}"
  fi
  echo "$MODEL" > "$MODEL_MARKER"
  echo "Download complete"
fi

# Generate a minimal config file so LocalAI can map the model name to the GGUF file.
# Without this, LocalAI looks for the model name as a filename (without .gguf extension).
GGUF_FILE=$(find /models -name "*.gguf" -type f | head -1)
if [[ -n "$GGUF_FILE" ]]; then
  GGUF_BASENAME=$(basename "$GGUF_FILE")
  MODEL_NAME="${GGUF_BASENAME%.gguf}"
  if [[ ! -f "/models/${MODEL_NAME}.yaml" ]]; then
    echo "Generating config for model: $MODEL_NAME -> $GGUF_BASENAME"
    cat > "/models/${MODEL_NAME}.yaml" <<CFGEOF
name: ${MODEL_NAME}
backend: llama-cpp
parameters:
  model: ${GGUF_BASENAME}
CFGEOF
  fi
fi
`
}

const runnerModelNameScript = `MODEL_NAME_SOURCE="${MODEL%%\#*}"
MODEL_NAME_SOURCE="${MODEL_NAME_SOURCE%%\?*}"
while [[ "$MODEL_NAME_SOURCE" == */ ]]; do
  MODEL_NAME_SOURCE="${MODEL_NAME_SOURCE%/}"
done
MODEL_NAME="${MODEL_NAME_SOURCE##*/}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Error: cannot derive a model name from '$MODEL'" >&2
  exit 1
fi`

// generateHFModelConfig generates the config logic for diffusers/vllm backends.
// These backends pass the HuggingFace model ID through to LocalAI and manage downloads themselves.
func generateHFModelConfig(backend string) string {
	return fmt.Sprintf(`# Check if model config matches the requested name, backend, and source (volume mount caching)
%[1]s
if [[ -f "/models/aikit-model.yaml" ]] &&
  grep -qxF "name: ${MODEL_NAME}" /models/aikit-model.yaml 2>/dev/null &&
  grep -qxF "backend: %[2]s" /models/aikit-model.yaml 2>/dev/null &&
  grep -qxF "  model: ${MODEL}" /models/aikit-model.yaml 2>/dev/null; then
  echo "Found existing %[2]s model config matching $MODEL in /models, skipping setup"
else
  if [[ -f "/models/aikit-model.yaml" ]]; then
    echo "Cached config does not match requested backend/model (%[2]s, $MODEL), regenerating"
  fi
  # For %[2]s backend, generate a LocalAI model config pointing to the HF model
  echo "Generating LocalAI config for %[2]s backend with model: $MODEL"
  cat > /models/aikit-model.yaml <<MODELEOF
name: ${MODEL_NAME}
backend: %[2]s
parameters:
  model: ${MODEL}
MODELEOF
  echo "Config generated at /models/aikit-model.yaml"
fi
`, runnerModelNameScript, backend)
}
