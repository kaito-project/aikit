---
title: Runner Images
---

Runner images are lightweight AIKit images that download models at container startup instead of embedding them at build time. This is useful when you want a single reusable image that can serve different models without rebuilding.

## Pre-built Runner Images

Pre-built runner images are available at `ghcr.io/kaito-project/aikit/runners/`:

| Image | Description |
|---|---|
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu` | CPU-only llama.cpp runner (amd64, arm64) |
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda` | CUDA + CPU fallback llama.cpp runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/diffusers-cuda` | CUDA diffusers runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cuda` | CUDA vLLM runner (amd64) |

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
```

:::tip
For HuggingFace repos with many quantization variants, use a **direct URL** to a specific file to avoid downloading all variants.
:::

Then query the model:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-1b-it-Q4_K_M", "messages": [{"role": "user", "content": "Hello!"}]}'
```

:::note
The model name in the API request is the GGUF filename without the `.gguf` extension.
:::

## GPU Support

CUDA runner images automatically detect whether an NVIDIA GPU is present at runtime. If no GPU is found, they fall back to CPU inference — no configuration needed.

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

Mount a volume to `/models` to cache downloaded models across container restarts:

```bash
docker run -v models:/models -p 8080:8080 \
  ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

The runner detects when a different model is requested and re-downloads automatically.

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

If you need a custom combination of backends or runtime configuration, you can build your own runner image. Define an aikitfile with `backends` but **no `models`**:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
backends:
  - llama-cpp
```

For CUDA:

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

### Supported Backends

| Backend | Description |
|---|---|
| `llama-cpp` | GGUF models via llama.cpp (CPU or CUDA) |
| `diffusers` | HuggingFace diffusers models (requires CUDA) |
| `vllm` | HuggingFace safetensors models via vLLM (requires CUDA) |
