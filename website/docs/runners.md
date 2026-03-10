---
title: Runner Images
---

Runner images are lightweight AIKit images that download models at container startup instead of embedding them at build time. This is useful when you want a single reusable image that can serve different models without rebuilding.

## Creating a Runner Image

Define an aikitfile with `backends` but **no `models`**:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
backends:
  - llama-cpp
```

Build it:

```bash
docker buildx build -t my-runner -f runner.yaml .
```

## Running with a Model

Pass the model reference as a container argument:

```bash
# Direct URL to a specific GGUF file (recommended for CI and reproducibility)
docker run -p 8080:8080 my-runner https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# HuggingFace repo (downloads all GGUF files in the repo)
docker run -p 8080:8080 my-runner unsloth/gemma-3-1b-it-GGUF

# With --model flag
docker run -p 8080:8080 my-runner --model unsloth/gemma-3-1b-it-GGUF
```

:::tip
For HuggingFace repos with many quantization variants, use a **direct URL** to a specific file to avoid downloading all variants.
:::

## Supported Backends

| Backend | Description |
|---|---|
| `llama-cpp` | GGUF models via llama.cpp (CPU or CUDA) |
| `diffusers` | HuggingFace diffusers models (requires CUDA) |
| `vllm` | HuggingFace safetensors models via vLLM (requires CUDA) |

## CUDA Runner Images

For GPU-accelerated inference, add `runtime: cuda`:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
runtime: cuda
backends:
  - llama-cpp
```

Run with GPU support:

```bash
docker run --gpus all -p 8080:8080 my-runner https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

:::note
CUDA runner images include a CPU fallback — if no NVIDIA GPU is detected at runtime, the image automatically uses the CPU backend.
:::

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace token for gated models |

```bash
docker run -e HF_TOKEN=hf_xxx -p 8080:8080 my-runner meta-llama/Llama-3.2-1B-Instruct-GGUF
```

## Volume Caching

Mount a volume to `/models` to cache downloaded models across container restarts:

```bash
docker run -v models:/models -p 8080:8080 my-runner https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
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

## Pre-built Runner Images

Pre-built runner images are available at `ghcr.io/kaito-project/aikit/runners/`:

| Image | Description |
|---|---|
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu` | CPU-only llama.cpp runner |
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda` | CUDA + CPU fallback llama.cpp runner |
| `ghcr.io/kaito-project/aikit/runners/diffusers-cuda` | CUDA diffusers runner |
| `ghcr.io/kaito-project/aikit/runners/vllm-cuda` | CUDA vLLM runner |
