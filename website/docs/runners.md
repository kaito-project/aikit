---
title: Runner Images
---

Runner images are reusable AIKit images that download models at runtime instead of embedding them at build time. This is useful when you want a single image that can serve different models without rebuilding.

Runner mode is available only for a resolved catalog entry with an explicit `runnerProfile`. The profile selects a reviewed AIKit adapter for model download, cache layout, input validation, generated configuration, and startup behavior. It is an internal catalog field, not a value users can add to an aikitfile.

A backend being installable in a standard model image does not make it runner-capable. When `backends` contains one family and `models` is empty, AIKit requests runner mode and fails if that exact family, runtime, and platform tuple has `runnerProfile: unsupported`. There is no silent switch back to standard mode or to a different runner.

## Pre-built Runner Images

Pre-built runner images are available at `ghcr.io/kaito-project/aikit/runners/`:

| Image | Description |
|---|---|
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cpu:latest` | CPU-only llama.cpp runner (amd64, arm64) |
| `ghcr.io/kaito-project/aikit/runners/llama-cpp-cuda:latest` | NVIDIA CUDA llama.cpp runner with a catalog-declared CPU companion backend (amd64) |
| `ghcr.io/kaito-project/aikit/runners/diffusers-cuda:latest` | NVIDIA CUDA diffusers runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cuda:latest` | NVIDIA CUDA vLLM runner (amd64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cpp-cpu:latest` | Experimental native vllm.cpp CPU runner (amd64, arm64) |
| `ghcr.io/kaito-project/aikit/runners/vllm-cpp-cuda:latest` | Experimental native vllm.cpp CUDA 13 runner for Blackwell GPUs (amd64) |

:::note
Pre-built runner images are currently published for CPU and NVIDIA CUDA only. The exact llama.cpp ROCm tuple is runner-capable for custom builds, but AIKit does not currently publish a pre-built ROCm runner image. Other ROCm families remain unavailable in runner mode unless their exact catalog entries name a runner profile.

Published image names describe the intended runner families, not every possible runtime. Consult the generated catalog lock for the selected frontend release before relying on a specific CUDA major, architecture, or experimental integration. NVIDIA L4T artifact selection for CUDA on Linux ARM64 is internal.
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

# Python vLLM with a persistent Hugging Face cache
docker run --gpus all -p 8080:8080 -v vllm-models:/models \
  ghcr.io/kaito-project/aikit/runners/vllm-cuda:latest \
  Qwen/Qwen2.5-0.5B-Instruct

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

# Python vLLM uses the repository's final path component as the model name
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

:::note
The model name in the API request is the GGUF filename without the `.gguf` extension. Diffusers, Python vLLM, and vllm.cpp repository sources use the repository name without the owner or pinned revision (for example, `Qwen2.5-0.5B-Instruct` or `Qwen3-0.6B`).
:::

:::warning
Runner mode downloads model content at container startup. A persistent cache reduces repeated transfers, but does not itself guarantee zero network access or verify the cached bytes as an immutable model artifact. A bare Hugging Face repository ID follows its upstream default revision; the Python vLLM runner's model argument does not provide an AIKit-validated immutable revision pin. Use a standard image with embedded model content for air-gapped or reproducible deployments.
:::

## GPU Support

The NVIDIA CUDA llama.cpp runner automatically detects whether an NVIDIA GPU is present at runtime. If no GPU is found, LocalAI can use the CPU companion artifact already declared and installed by that CUDA catalog plan. This is runtime behavior inside one selected plan, not catalog resolution falling back to a CPU tuple. The Diffusers and Python vLLM (`vllm`) runners require an NVIDIA GPU. ROCm runner images are not published, but a custom llama.cpp ROCm runner can be built from the catalog tuple.

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

For AMD ROCm on `linux/amd64`:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
runtime: rocm
backends:
  - llama-cpp
```

Build the ROCm runner for its exact catalog platform:

```bash
docker buildx build --platform linux/amd64 -t my-rocm-runner -f runner.yaml .
```

Build the CPU or NVIDIA examples for the desired supported platform:

```bash
docker buildx build -t my-runner -f runner.yaml .
```

Each example is a catalog request, not an open-ended backend download. The build succeeds only when the exact entry is selectable (`supported` or `experimental`) and its `runnerProfile` is not `unsupported`. Missing tuples and `quarantined` or `deprecated` entries fail the build; AIKit does not fall back to another runner, family, CUDA major, runtime, or platform. Use `runtime` to request the execution profile, as described in the [Inference API Specifications](specs-inference.md#backend-catalog-selection).

### Explicit runner adapters

AIKit implements the following runner adapter behaviors. The embedded catalog decides which exact tuples may use them; the family names below describe the current assignments, not a general promise for every runtime or platform.

| `runnerProfile` | Current family assignment | Behavior |
|---|---|---|
| `llama-cpp` | `llama-cpp` | GGUF repositories or direct HTTP(S) GGUF files, with a runner-owned model cache. |
| `hf-config` | `diffusers`, `vllm` | Generates a LocalAI config that lets the backend use its Hugging Face integration. |
| `vllm-cpp` | `vllm-cpp` | Direct HTTP(S) GGUF files or Hugging Face safetensors repositories; pin a repository commit for reproducibility. |

A family without a runner adapter can still be installed in standard mode when its exact catalog tuple is selectable, but it cannot enter runner mode until a frontend release explicitly assigns and implements a suitable `runnerProfile`.
