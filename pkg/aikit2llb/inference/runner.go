package inference

import (
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	runnerHuggingFaceHubVersion = "1.26.0"
	runnerConfigDir             = "/models/aikit-runner"
	runnerConfigFilename        = "model.yaml"
	runnerConfigPath            = runnerConfigDir + "/" + runnerConfigFilename
	runnerLegacyModelMarker     = "/models/.aikit-model-ref"
	runnerLlamaCppModelDir      = "/models/llama-cpp-model"
	runnerLlamaCppModelMarker   = "/models/.aikit-llama-cpp-model-ref"
	runnerVLLMCppModelDir       = "/models/vllm-cpp-model"
	runnerVLLMCppModelMarker    = "/models/.aikit-vllm-cpp-model-ref"
	runnerAPTSetupCommand       = "if [ -f /etc/apt/sources.list ]; then sed -i 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g; " +
		"s|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; fi && " +
		"if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then sed -i 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g; " +
		"s|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; fi && " +
		"apt-get -o Acquire::Retries=5 -o APT::Update::Error-Mode=any update && "
	runnerDependenciesCleanup = "apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /root/.cache/pip"
	runnerDownloaderPackages  = "curl ca-certificates python3 python3-pip"
	runnerTrustStorePackage   = "ca-certificates"
	runnerHFCLIInstallCommand = "(pip install --no-cache-dir --no-compile --break-system-packages huggingface-hub==" + runnerHuggingFaceHubVersion + " 2>/dev/null || " +
		"pip install --no-cache-dir --no-compile huggingface-hub==" + runnerHuggingFaceHubVersion + ")"
)

// isRunnerMode returns true when the config defines backends but no models,
// indicating a "runner" image that resolves models at runtime.
func isRunnerMode(c *config.InferenceConfig) bool {
	return len(c.Backends) > 0 && len(c.Models) == 0
}

func installRunnerDependenciesWithBackend(backend backendcatalog.Resolution, s llb.State, merge llb.State, platform specs.Platform) (llb.State, llb.State) {
	command := runnerDependenciesCommand(backend.RunnerProfile)
	if command == "" {
		return s, merge
	}

	savedState := s

	// Install a trust store for every runtime downloader and add the hf CLI only for native model payloads.
	// Note: Runner mode is not supported for Apple Silicon (validated in build).
	s = s.Run(
		utils.Sh(command),
		llb.WithCustomNamef("Installing runner dependencies for platform %s/%s", platform.OS, platform.Architecture),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

func runnerDependenciesCommand(profile backendcatalog.RunnerProfile) string {
	packages := runnerTrustStorePackage
	installHFCLI := false

	switch profile {
	case backendcatalog.RunnerProfileHFConfig:
	case backendcatalog.RunnerProfileLlamaCpp, backendcatalog.RunnerProfileVLLMCpp:
		packages = runnerDownloaderPackages
		installHFCLI = true
	default:
		return ""
	}

	command := runnerAPTSetupCommand + "apt-get install --no-install-recommends -y " + packages + " && "
	if installHFCLI {
		command += runnerHFCLIInstallCommand + " && "
	}

	return command + runnerDependenciesCleanup
}

// installRunnerEntrypoint writes the entrypoint script and creates the /models/
// directory with correct ownership for non-root compatibility.
func installRunnerEntrypoint(c *config.InferenceConfig, backend backendcatalog.Resolution, s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s

	script := generateRunnerScriptWithBackend(c, backend)

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
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	if c.Runtime == utils.RuntimeAppleSilicon {
		platform.Architecture = utils.PlatformARM64
	}
	backend, err := ResolveBackend(c, platform)
	if err != nil {
		panic("resolving backend for runner script: " + err.Error())
	}

	return generateRunnerScriptWithBackend(c, backend)
}

func generateRunnerScriptWithBackend(c *config.InferenceConfig, backend backendcatalog.Resolution) string {
	backendName := backend.Family
	configDirectory := runnerConfigDir
	switch backend.RunnerProfile {
	case backendcatalog.RunnerProfileLlamaCpp:
		configDirectory = runnerLlamaCppModelDir
	case backendcatalog.RunnerProfileVLLMCpp:
		configDirectory = runnerVLLMCppModelDir
	}

	var sb strings.Builder
	sb.WriteString(`#!/bin/bash
set -euo pipefail

BACKEND="` + backendName + `"
RUNNER_PROFILE="` + string(backend.RunnerProfile) + `"

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
  if [[ "$RUNNER_PROFILE" == "vllm-cpp" ]]; then
    echo "  docker run -p 8080:8080 <image> org/safetensors-model@0123456789abcdef0123456789abcdef01234567"
    echo "  docker run -p 8080:8080 <image> https://example.com/model.gguf"
    echo "  docker run -p 8080:8080 <image> --model org/safetensors-model"
  else
    echo "  docker run -p 8080:8080 <image> org/model"
    echo "  docker run -p 8080:8080 <image> https://example.com/model.gguf"
    echo "  docker run -p 8080:8080 <image> --model org/model"
  fi
  echo ""
  echo "Environment variables:"
  echo "  HF_TOKEN    - HuggingFace token for gated models"
  exit 1
fi

# Keep generated configs and payloads inside backend-owned directories. Native
# backends scan their cache directory because LocalAI requires model paths to be
# relative to --models-path; other backends use the config-only directory.
RUNNER_CONFIG_DIR="` + configDirectory + `"
RUNNER_CONFIG="$RUNNER_CONFIG_DIR/` + runnerConfigFilename + `"

# Strip URI scheme prefixes (e.g. huggingface://org/repo -> org/repo)
# kubeairunway passes model IDs with the huggingface:// prefix.
MODEL="${MODEL#huggingface://}"

# Cache markers contain only a digest so authenticated URLs are never persisted.
# Logs omit URL queries/fragments and redact URI userinfo for the same reason.
MODEL_CACHE_KEY="$(printf '%s' "$MODEL" | sha256sum)"
MODEL_CACHE_KEY="${MODEL_CACHE_KEY%% *}"
MODEL_SCHEME=""
if [[ "$MODEL" == *://* ]]; then
  MODEL_SCHEME="${MODEL%%://*}"
  MODEL_SCHEME="$(printf '%s' "$MODEL_SCHEME" | tr '[:upper:]' '[:lower:]')"
fi
MODEL_LOG_REF="$MODEL"
if [[ -n "$MODEL_SCHEME" ]]; then
  MODEL_LOG_REF="${MODEL_LOG_REF%%\#*}"
  MODEL_LOG_REF="${MODEL_LOG_REF%%\?*}"
  MODEL_LOG_REF="${MODEL_LOG_REF#*://}"
  if [[ "$MODEL_LOG_REF" == *@* ]]; then
    MODEL_LOG_REF="[redacted]@${MODEL_LOG_REF##*@}"
  fi
  MODEL_LOG_REF="${MODEL_SCHEME}://${MODEL_LOG_REF}"
fi

echo "AIKit Runner: backend=$BACKEND model=$MODEL_LOG_REF"

`)

	// Backend-specific download logic
	switch backend.RunnerProfile {
	case backendcatalog.RunnerProfileLlamaCpp:
		sb.WriteString(generateLlamaCppDownload())
	case backendcatalog.RunnerProfileVLLMCpp:
		sb.WriteString(generateVLLMCppDownload())
	case backendcatalog.RunnerProfileHFConfig:
		sb.WriteString(generateHFModelConfig(backendName))
	default:
		panic(fmt.Sprintf("unsupported validated runner profile %q", backend.RunnerProfile))
	}

	// Start LocalAI
	sb.WriteString(`
# Start local-ai
LOCAL_AI_ARGS=("--models-path" "$RUNNER_CONFIG_DIR")
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
MODEL_MARKER="` + runnerLlamaCppModelMarker + `"
LEGACY_MODEL_MARKER="` + runnerLegacyModelMarker + `"
LLAMA_CPP_MODEL_DIR="` + runnerLlamaCppModelDir + `"
LLAMA_CPP_PAYLOAD_DIR="$LLAMA_CPP_MODEL_DIR/payload"
CACHED_GGUF=$(find "$LLAMA_CPP_PAYLOAD_DIR" -type f -name "*.gguf" -print -quit 2>/dev/null || true)
if [[ -f "$MODEL_MARKER" ]] && [[ "$(cat "$MODEL_MARKER")" == "$MODEL_CACHE_KEY" ]] &&
  [[ -n "$CACHED_GGUF" ]]; then
  echo "Found cached model matching $MODEL_LOG_REF in $LLAMA_CPP_MODEL_DIR, skipping download"
else
  # Different model requested, missing payload, or a legacy marker — clean and
  # re-download only the runner-owned cache. Legacy root files have no reliable
  # ownership metadata, so leave them untouched and outside LocalAI's config path.
  if [[ -f "$MODEL_MARKER" ]]; then
    echo "Cached model data is missing or does not match requested model ($MODEL_LOG_REF), re-downloading"
  elif [[ -f "$LEGACY_MODEL_MARKER" ]]; then
    echo "Migrating legacy cached model for requested model ($MODEL_LOG_REF)"
  fi
  rm -rf "$LLAMA_CPP_MODEL_DIR"
  rm -f "$MODEL_MARKER" "$LEGACY_MODEL_MARKER" "$RUNNER_CONFIG"
  mkdir -p "$LLAMA_CPP_PAYLOAD_DIR"
  if [[ "$MODEL_SCHEME" == "http" ]] || [[ "$MODEL_SCHEME" == "https" ]]; then
    # Direct HTTP/HTTPS download
    echo "Downloading model from URL: $MODEL_LOG_REF"
    MODEL_URL_PATH="${MODEL%%\#*}"
    MODEL_URL_PATH="${MODEL_URL_PATH%%\?*}"
    FILENAME="${MODEL_URL_PATH##*/}"
    if [[ -z "$FILENAME" ]] || [[ "$FILENAME" == "." ]] || [[ "$FILENAME" == ".." ]]; then
      echo "Error: cannot derive a model filename from $MODEL_LOG_REF" >&2
      exit 1
    fi
    curl -fL --progress-bar -o "$LLAMA_CPP_PAYLOAD_DIR/$FILENAME" "$MODEL"
  else
    # HuggingFace repo - download GGUF files
    echo "Downloading GGUF files from HuggingFace: $MODEL_LOG_REF"
    HF_ARGS=("$MODEL" "--local-dir" "$LLAMA_CPP_PAYLOAD_DIR" "--include" "*.gguf")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_ARGS+=("--token" "$HF_TOKEN")
    fi
    hf download "${HF_ARGS[@]}"
  fi
  DOWNLOADED_GGUF=$(find "$LLAMA_CPP_PAYLOAD_DIR" -type f -name "*.gguf" -print -quit)
  if [[ -z "$DOWNLOADED_GGUF" ]]; then
    echo "Error: no GGUF file was downloaded for $MODEL_LOG_REF" >&2
    exit 1
  fi
  printf '%s\n' "$MODEL_CACHE_KEY" > "$MODEL_MARKER"
  echo "Download complete"
fi

# Generate a minimal config file so LocalAI can map the model name to the GGUF file.
# Without this, LocalAI looks for the model name as a filename (without .gguf extension).
GGUF_FILE=$(find "$LLAMA_CPP_PAYLOAD_DIR" -type f -name "*.gguf" -print -quit)
if [[ -z "$GGUF_FILE" ]]; then
  echo "Error: cached GGUF file for $MODEL_LOG_REF is missing" >&2
  exit 1
fi
GGUF_BASENAME=$(basename "$GGUF_FILE")
MODEL_NAME="${GGUF_BASENAME%.gguf}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Error: cannot derive a LocalAI model name from $GGUF_BASENAME" >&2
  exit 1
fi
MODEL_RELATIVE_PATH="${GGUF_FILE#"$LLAMA_CPP_MODEL_DIR"/}"
if [[ -z "$MODEL_RELATIVE_PATH" ]] || [[ "$MODEL_RELATIVE_PATH" == "$GGUF_FILE" ]]; then
  echo "Error: cached GGUF file is outside $LLAMA_CPP_MODEL_DIR" >&2
  exit 1
fi
YAML_MODEL_NAME=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$MODEL_NAME")
YAML_MODEL_PATH=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$MODEL_RELATIVE_PATH")
mkdir -p "$RUNNER_CONFIG_DIR"
echo "Generating config for model: $MODEL_NAME -> $GGUF_FILE"
CONFIG_TMP=$(mktemp "$RUNNER_CONFIG_DIR/.model.yaml.XXXXXX")
cat > "$CONFIG_TMP" <<CFGEOF
name: ${YAML_MODEL_NAME}
backend: llama-cpp
parameters:
  model: ${YAML_MODEL_PATH}
CFGEOF
mv "$CONFIG_TMP" "$RUNNER_CONFIG"
`
}

// generateVLLMCppDownload generates download and configuration logic for the
// vllm.cpp backend. Unlike the Python vLLM backend, vllm.cpp requires a local
// model path rather than a Hugging Face repository ID.
func generateVLLMCppDownload() string {
	return `# vllm.cpp requires the model to be materialized locally before startup.
MODEL_MARKER="` + runnerVLLMCppModelMarker + `"
VLLM_CPP_MODEL_DIR="` + runnerVLLMCppModelDir + `"
VLLM_CPP_PAYLOAD_DIR="$VLLM_CPP_MODEL_DIR/payload"
MODEL_KIND=""
LOCAL_MODEL_PATH=""
CONFIG_MODEL_PATH=""
HF_REPOSITORY=""
HF_REVISION=""

validate_vllm_cpp_repository() {
  [[ -f "$VLLM_CPP_PAYLOAD_DIR/config.json" ]] || return 1
  [[ -f "$VLLM_CPP_PAYLOAD_DIR/tokenizer.json" ]] || return 1

  local index_path="$VLLM_CPP_PAYLOAD_DIR/model.safetensors.index.json"
  if [[ -f "$index_path" ]]; then
    python3 - "$VLLM_CPP_PAYLOAD_DIR" "$index_path" <<'PYEOF'
import json
import os
import sys

root = os.path.realpath(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as index_file:
    weight_map = json.load(index_file).get("weight_map")

if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(1)

for relative_path in set(weight_map.values()):
    if (
        not isinstance(relative_path, str)
        or not relative_path.endswith(".safetensors")
        or os.path.basename(relative_path) != relative_path
    ):
        raise SystemExit(1)
    candidate = os.path.join(root, relative_path)
    if not os.path.isfile(candidate):
        raise SystemExit(1)
PYEOF
    return
  fi

  [[ -n "$(find "$VLLM_CPP_PAYLOAD_DIR" -maxdepth 1 -type f -name "*.safetensors" -print -quit)" ]]
}

case "$MODEL_SCHEME" in
  http|https)
    # Direct URLs are supported only for a single GGUF file. Strip query and
    # fragment components before validating and deriving the local filename.
    MODEL_URL_PATH="${MODEL%%\#*}"
    MODEL_URL_PATH="${MODEL_URL_PATH%%\?*}"
    GGUF_FILENAME="${MODEL_URL_PATH##*/}"
    if [[ ! "$GGUF_FILENAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$ ]]; then
      echo "Error: vllm-cpp direct URLs must point to a .gguf file with a safe filename" >&2
      exit 1
    fi
    MODEL_KIND="gguf"
    MODEL_NAME="${GGUF_FILENAME%.gguf}"
    LOCAL_MODEL_PATH="${VLLM_CPP_PAYLOAD_DIR}/${GGUF_FILENAME}"
    CONFIG_MODEL_PATH="payload/$GGUF_FILENAME"
    ;;
  ?*)
    echo "Error: vllm-cpp supports only huggingface:// repository references or HTTP(S) .gguf URLs" >&2
    exit 1
    ;;
  "")
    # Accept an optional immutable Hugging Face commit after @. Restrict both
    # repository IDs and revisions before passing them to the hf CLI.
    HF_REPOSITORY="${MODEL%%@*}"
    if [[ "$MODEL" == *@* ]]; then
      HF_REVISION="${MODEL#*@}"
      if [[ ! "$HF_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
        echo "Error: Hugging Face revisions for vllm-cpp must be 40-character lowercase commit SHAs" >&2
        exit 1
      fi
    fi
    if [[ ! "$HF_REPOSITORY" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$ ]]; then
      echo "Error: invalid Hugging Face repository ID for vllm-cpp: $MODEL" >&2
      exit 1
    fi
    MODEL_KIND="repository"
    MODEL_NAME="${HF_REPOSITORY##*/}"
    LOCAL_MODEL_PATH="$VLLM_CPP_PAYLOAD_DIR"
    CONFIG_MODEL_PATH="payload"
    ;;
esac

# Reuse a materialized model only when both its source marker and local payload
# match. This keeps a mounted /models volume model-aware across runner restarts.
CACHE_HIT="false"
if [[ -f "$MODEL_MARKER" ]] && [[ "$(cat "$MODEL_MARKER")" == "$MODEL_CACHE_KEY" ]]; then
  if [[ "$MODEL_KIND" == "gguf" ]] && [[ -f "$LOCAL_MODEL_PATH" ]]; then
    CACHE_HIT="true"
  elif [[ "$MODEL_KIND" == "repository" ]] && validate_vllm_cpp_repository; then
    CACHE_HIT="true"
  fi
fi

if [[ "$CACHE_HIT" == "true" ]]; then
  echo "Found cached vllm-cpp model matching $MODEL_LOG_REF in $VLLM_CPP_MODEL_DIR, skipping download"
else
  if [[ -f "$MODEL_MARKER" ]]; then
    echo "Cached model does not match requested vllm-cpp model ($MODEL_LOG_REF), re-downloading"
  fi

  # This fixed runner-owned directory avoids deriving filesystem paths from
  # untrusted model references and makes cache replacement deterministic.
  rm -rf ` + runnerVLLMCppModelDir + `
  rm -f "$MODEL_MARKER" "$RUNNER_CONFIG"
  mkdir -p "$VLLM_CPP_PAYLOAD_DIR"

  if [[ "$MODEL_KIND" == "gguf" ]]; then
    echo "Downloading vllm-cpp GGUF model from URL: $MODEL_LOG_REF"
    PARTIAL_PATH="${LOCAL_MODEL_PATH}.part"
    curl -fL --progress-bar --output "$PARTIAL_PATH" "$MODEL"
    mv "$PARTIAL_PATH" "$LOCAL_MODEL_PATH"
  else
    echo "Downloading Hugging Face repository for vllm-cpp: $MODEL_LOG_REF"
    # Download only the native vllm.cpp weight, configuration, and tokenizer
    # assets. This avoids materializing alternate PyTorch, ONNX, Flax, or GGUF
    # weights that can coexist in the same repository.
    HF_ARGS=(
      "$HF_REPOSITORY"
      "--local-dir" "$VLLM_CPP_PAYLOAD_DIR"
      "--include" "*.json"
      "--include" "*.safetensors"
      "--include" "*.model"
      "--include" "*.txt"
      "--include" "*.tiktoken"
      "--include" "*.jinja"
      "--exclude" "*.gguf"
    )
    if [[ -n "$HF_REVISION" ]]; then
      HF_ARGS+=("--revision" "$HF_REVISION")
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_ARGS+=("--token" "$HF_TOKEN")
    fi
    hf download "${HF_ARGS[@]}"

    if ! validate_vllm_cpp_repository; then
      echo "Error: vllm-cpp Hugging Face repositories must contain config.json and safetensors weights, plus tokenizer.json; use a direct HTTP(S) .gguf URL for GGUF models" >&2
      exit 1
    fi
  fi

  printf '%s\n' "$MODEL_CACHE_KEY" > "$MODEL_MARKER"
  echo "Download complete"
fi

# Always point LocalAI at the validated local payload. A bare Hugging Face ID
# is valid for the Python vLLM backend but is not valid for vllm.cpp.
mkdir -p "$RUNNER_CONFIG_DIR"
CONFIG_TMP=$(mktemp "$RUNNER_CONFIG_DIR/.model.yaml.XXXXXX")
cat > "$CONFIG_TMP" <<MODELEOF
name: '${MODEL_NAME}'
backend: vllm-cpp
parameters:
  model: '${CONFIG_MODEL_PATH}'
template:
  use_tokenizer_template: true
MODELEOF
mv "$CONFIG_TMP" "$RUNNER_CONFIG"
echo "Config generated at $RUNNER_CONFIG"
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
if [[ -f "$RUNNER_CONFIG" ]] &&
  grep -qxF "name: ${MODEL_NAME}" "$RUNNER_CONFIG" 2>/dev/null &&
  grep -qxF "backend: %[2]s" "$RUNNER_CONFIG" 2>/dev/null &&
  grep -qxF "  model: ${MODEL}" "$RUNNER_CONFIG" 2>/dev/null; then
  echo "Found existing %[2]s model config matching $MODEL in /models, skipping setup"
else
  if [[ -f "$RUNNER_CONFIG" ]]; then
    echo "Cached config does not match requested backend/model (%[2]s, $MODEL), regenerating"
  fi
  # For %[2]s backend, generate a LocalAI model config pointing to the HF model
  echo "Generating LocalAI config for %[2]s backend with model: $MODEL"
  mkdir -p "$RUNNER_CONFIG_DIR"
  CONFIG_TMP=$(mktemp "$RUNNER_CONFIG_DIR/.model.yaml.XXXXXX")
  cat > "$CONFIG_TMP" <<MODELEOF
name: ${MODEL_NAME}
backend: %[2]s
parameters:
  model: ${MODEL}
MODELEOF
  mv "$CONFIG_TMP" "$RUNNER_CONFIG"
  echo "Config generated at $RUNNER_CONFIG"
fi
`, runnerModelNameScript, backend)
}
