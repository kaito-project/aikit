---
title: Runner Images
---

Runner images are reusable AIKit images that download models at runtime instead of embedding them at build time. This is useful when you want a single image that can serve different models without rebuilding.

Runner mode is available only for backend catalog entries that name an audited runner profile. A backend being usable in a model image does not by itself mean its runtime downloader, cache layout, input validation, and startup behavior have been audited for runner mode. The catalog embedded in the selected frontend release is the authority for the exact backend, runtime, capability, and platform tuple.

## Pre-built Runner Images

Pre-built runner images are available at `ghcr.io/kaito-project/aikit/runners/`:

| Image | Description |
|---|---|
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest` | CPU-only llama.cpp runner (amd64, arm64) |
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda:latest` | NVIDIA CUDA + CPU fallback llama.cpp runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/diffusers-cuda:latest` | NVIDIA CUDA diffusers runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cuda:latest` | NVIDIA CUDA vLLM runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cpp-cpu:latest` | Experimental native vllm.cpp CPU runner (amd64, arm64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cpp-cuda:latest` | Experimental native vllm.cpp CUDA 13 runner for Blackwell GPUs (amd64) |

:::note
Pre-built runner images are currently published for CPU and NVIDIA CUDA only. ROCm catalog entries do not yet have an audited runner profile, so runner-mode builds for AMD GPUs fail closed.

Published image names describe the intended runner families, not every possible backend capability. Consult the generated catalog lock for the selected frontend release before relying on a specific CUDA major, L4T variant, architecture, or experimental integration.
:::

## Quick Start

Pass a model reference as a container argument:

```bash
# Direct URL to a specific GGUF file (recommended)
docker run -p 8080:8080 ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# HuggingFace repo (downloads all GGUF files in the repo)
docker run -p 8080:8080 ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest \
  unsloth/gemma-3-1b-it-GGUF

# With GPU support
docker run --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# Native vllm.cpp with a safetensors repository pinned to an immutable revision
docker run -p 8080:8080 ghcr.io/kaito-project/aikit/runners/vllm-cpp-cpu:latest \
  Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
```

:::tip
For GGUF repositories with many quantization variants, use a **direct URL** to a specific `.gguf` file to avoid downloading all variants. The vllm.cpp runner accepts safetensors only as a repository reference; use `owner/repository@commit` with the full 40-character lowercase commit SHA for reproducible downloads.
:::

Then query the model:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-1b-it-Q4_K_M", "messages": [{"role": "user", "content": "Hello!"}]}'
```

:::note
The model name in the API request is the GGUF filename without the `.gguf` extension. For a vllm.cpp safetensors repository, it is the repository name without the owner or pinned revision (for example, `Qwen3-0.6B`).
:::

## GPU Support

The NVIDIA CUDA llama.cpp runner automatically detects whether an NVIDIA GPU is present at runtime. If no GPU is found, it falls back to CPU inference — no configuration needed. The Diffusers and vLLM runners require an NVIDIA GPU. ROCm runner images are not published yet.

```bash
# With GPU
docker run --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# Same image works without GPU (automatically uses CPU)
docker run -p 8080:8080 ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace token for gated models |

```bash
docker run -e HF_TOKEN=hf_xxx -p 8080:8080 \
  ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest \
  meta-llama/Llama-3.2-1B-Instruct-GGUF
```

## Volume Caching

Mount a volume to `/models` to cache downloaded models across container restarts. The llama.cpp and vllm.cpp runners keep their payloads in `/models/llama-cpp-model` and `/models/vllm-cpp-model`, respectively. Diffusers and Python vLLM store their Hugging Face cache under `/models/.cache/huggingface`:

```bash
docker run -v models:/models -p 8080:8080 \
  ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

Native runners detect when a different model is requested and replace only their backend-owned cache. Their generated configs live at `/models/llama-cpp-model/model.yaml` and `/models/vllm-cpp-model/model.yaml`; downloaded payloads stay under each directory's unscanned `payload/` subtree. Diffusers and Python vLLM use `/models/aikit-runner/model.yaml` while retaining the shared Hugging Face cache for reuse. LocalAI scans only the active runner's config directory, not unrelated YAML elsewhere in the mounted volume.

## Kubernetes / kubeairunway

Runner images are compatible with [kubeairunway](https://github.com/kaito-project/kubeairunway). The `huggingface://` URI scheme used by kubeairunway is automatically handled:

```yaml
apiVersion: kubeairunway.ai/v1alpha1
kind: ModelDeployment
metadata:
  name: gemma-cpu
spec:
  model:
    id: "google/gemma-3-1b-it-qat-q8_0-gguf"
    source: huggingface
  engine:
    type: llamacpp
  image: "ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest"
```

## Building Custom Runner Images

If you need a custom backend or runtime configuration, you can request a custom runner image. Define an aikitfile with `backends` but **no `models`**:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
backends:
  - llama-cpp
```

For NVIDIA CUDA:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
runtime: cuda
backends:
  - llama-cpp
```

Build:

```bash
docker buildx build -t my-runner -f runner.yaml .
```

Each example is a catalog request, not an open-ended backend download. The build succeeds only when the exact tuple has an audited runner profile with `supported` or `experimental` status. Missing, `quarantined`, and `deprecated` profiles fail the build; AIKit does not fall back to another runner or backend. Use `backendCapability` when an audited profile requires an exact selector, as described in the [Inference API Specifications](specs-inference.md#backend-catalog-selection).

### Runner-capable backend families

This table describes the runner interface implemented by each backend family. Actual release availability is limited to the audited profiles in the embedded catalog.

| Backend | Description |
|---|---|
| `llama-cpp` | GGUF models via llama.cpp (CPU or NVIDIA CUDA; ROCm is not runner-enabled in the current catalog) |
| `diffusers` | HuggingFace diffusers models (requires NVIDIA CUDA) |
| `vllm` | HuggingFace safetensors models via vLLM (requires NVIDIA CUDA) |
| `vllm-cpp` | Direct HTTP(S) GGUF URLs or Hugging Face safetensors repositories via the experimental native engine (CPU, or CUDA 13 on amd64 Blackwell GPUs) |
