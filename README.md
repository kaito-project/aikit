# AIKit ✨

<p align="center">
<img src="./website/static/img/logo.png" width="200"><br>
</p>

AIKit is a comprehensive platform to quickly get started to host, deploy, build and fine-tune large language models (LLMs).

AIKit offers three main capabilities:

- **Inference**: AIKit uses [LocalAI](https://localai.io/), which supports a wide range of inference capabilities and formats. LocalAI provides a drop-in replacement REST API that is OpenAI API compatible, so you can use any OpenAI API compatible client, such as [Kubectl AI](https://github.com/sozercan/kubectl-ai), [Chatbot-UI](https://github.com/sozercan/chatbot-ui) and many more, to send requests to open LLMs!

- **[Fine-Tuning](https://kaito-project.github.io/aikit/docs/fine-tune)**: AIKit offers an extensible fine-tuning interface. It supports [Unsloth](https://github.com/unslothai/unsloth) for fast, memory efficient, and easy fine-tuning experience.

- **[OCI Packaging](https://kaito-project.github.io/aikit/docs/packaging)**: Package models as OCI artifacts for distribution through any OCI-compliant registry. Supports [CNCF ModelPack](https://github.com/modelpack/model-spec) specification and generic artifact packaging.

👉 For full documentation, please see [AIKit website](https://kaito-project.github.io/aikit/)!

## Features

- 🐳 Run GGUF text models on a CPU with [Docker](https://docs.docker.com/desktop/install/linux-install/) or [Podman](https://podman.io), including offline inference after pulling the image.
- 🤏 Minimal image size, resulting in less vulnerabilities and smaller attack surface with a custom [chiseled](https://ubuntu.com/containers/chiseled) image
- 🎵 [Fine-tune support](https://kaito-project.github.io/aikit/docs/fine-tune)
- 📦 [OCI packaging support](https://kaito-project.github.io/aikit/docs/packaging) for distributing models as OCI artifacts
- 🚀 Easy to use declarative configuration for [inference](https://kaito-project.github.io/aikit/docs/specs-inference) and [fine-tuning](https://kaito-project.github.io/aikit/docs/specs-finetune)
- ✨ OpenAI API compatible to use with any OpenAI API compatible client
- 📸 [Multi-modal model support](https://kaito-project.github.io/aikit/docs/vision)
- 🖼️ [Image generation support](https://kaito-project.github.io/aikit/docs/diffusion)
- 🦙 Support for GGUF ([`llama`](https://github.com/ggerganov/llama.cpp)) and GGML ([`llama-ggml`](https://github.com/ggerganov/llama.cpp)) models
- 🚢 [Kubernetes deployment ready](https://kaito-project.github.io/aikit/docs/kubernetes)
- 📚 Supports multiple models with a single image
- 🖥️ Supports [AMD64 and ARM64](https://kaito-project.github.io/aikit/docs/create-images#multi-platform-support) CPUs and [GPU-accelerated inferencing with NVIDIA CUDA and AMD ROCm support](https://kaito-project.github.io/aikit/docs/gpu)
- 🔐 Ensure [supply chain security](https://kaito-project.github.io/aikit/docs/security) with SBOMs, Provenance attestations, and signed images
- 🌈 Supports air-gapped inference with self-hosted or local registries when model content and dependencies are baked or mirrored ahead of time; runner images that download models at startup are not air-gapped by default.

## Quick start

Start with Qwen 3.5 4B for text chat. Its Q4_K_M weights are about 2.74 GB; allow additional memory for the runtime and context.

> [!NOTE]
> The new Qwen, Gemma 4, Devstral Small 2, and FLUX.2 images are pending staging validation and publication. The commands below use their planned tags. Until publication, use `ghcr.io/kaito-project/aikit/llama3.2:3b` with API model ID `llama-3.2-3b-instruct`. See the [model release process](./website/docs/release.md#predefined-models).

```bash
docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b
```

Open [http://localhost:8080/chat](http://localhost:8080/chat) for the WebUI, or call the OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "qwen-3.5-4b",
    "messages": [{"role": "user", "content": "explain kubernetes in a sentence"}]
  }'
```

## Pre-made models

The recommended presets are listed below. Image names use the prefix `ghcr.io/kaito-project/aikit/`. All recommended weights use Apache 2.0 licenses; the model links provide upstream details.

Gemma 4 uses Google's [Apache 2.0 license](https://ai.google.dev/gemma/apache_2).

| Model | Image | API model ID | GGUF size |
| --- | --- | --- | --- |
| [Qwen 3.5 2B](https://huggingface.co/Qwen/Qwen3.5-2B) | `qwen3.5:2b` | `qwen-3.5-2b` | 1.28 GB |
| [Qwen 3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B), quickstart | `qwen3.5:4b` | `qwen-3.5-4b` | 2.74 GB |
| [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `qwen3.5:9b` | `qwen-3.5-9b` | 5.68 GB |
| [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3.8:27b` | `qwen-3.8-27b` | 16.46 GB |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | `gemma4:e2b` | `gemma-4-e2b-instruct` | 3.35 GB |
| [Devstral Small 2 24B](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512), coding | `devstral-small2:24b` | `devstral-small-2-24b-instruct` | 14.33 GB |
| [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) | `gpt-oss:20b` | `gpt-oss-20b` | 12.11 GB |
| [GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b) | `gpt-oss:120b` | `gpt-oss-120b` | 63.39 GB |
| [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), image generation | `flux2:klein-4b` | `flux-2-klein-4b` | Downloads on first use |

GGUF sizes are weight downloads, not runtime memory requirements or complete image sizes. The text presets include no multimodal projectors. Gemma's E2B label describes effective parameters; its full model including embeddings is about 5.1B parameters.

Llama 3.2 1B/3B, Llama 3.1 8B, Llama 3.3 70B, and Phi 4 14B remain maintained compatibility options with their existing image tags and API IDs. Gemma 2, Mixtral 8x7B, QwQ 32B, Codestral 22B, and FLUX.1 Dev are retired from publishing and weekly patching. See [pre-made models](https://kaito-project.github.io/aikit/docs/premade-models) for compatibility tags and migration guidance.

### CPU and NVIDIA CUDA

Text images support AMD64 and ARM64 and include a CPU backend. Docker selects the image for your architecture. To use an NVIDIA GPU, add `--gpus all`:

```bash
docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b
```

FLUX.2 uses the experimental CUDA 12 backend plan on AMD64 and downloads weights from Hugging Face on first use. See [image generation requirements](https://kaito-project.github.io/aikit/docs/premade-models#image-generation).

### Apple Silicon, experimental

GGUF text presets also have an experimental Apple Silicon image path:

```bash
podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.5:4b
```

Set up GPU access using the [Podman Desktop instructions](https://podman-desktop.io/docs/podman/gpu). This profile targets Apple Silicon and does not support Intel Macs or FLUX.2.

### AMD ROCm, experimental

For AMD GPUs, [create a custom image](https://kaito-project.github.io/aikit/docs/create-images) using `llama-cpp` with `runtime: rocm` on `linux/amd64`. Follow the device setup in [GPU acceleration](https://kaito-project.github.io/aikit/docs/gpu). Pre-made ROCm images are not published.

## Contributing

Want to contribute to AIKit? Check out our [Contributing Guide](./CONTRIBUTING.md) for development setup, testing instructions, and contribution guidelines.

## What's next?

👉 For more information and how to fine tune models or create your own images, please see [AIKit website](https://kaito-project.github.io/aikit/)!
