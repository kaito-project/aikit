# AIKit ✨

<p align="center">
<img src="./website/static/img/logo.png" width="200"><br>
</p>

AIKit is a comprehensive platform to quickly get started to host, deploy, build and fine-tune large language models (LLMs).

AIKit offers two main capabilities:

- **Inference**: AIKit uses [LocalAI](https://localai.io/), which supports a wide range of inference capabilities and formats. LocalAI provides a drop-in replacement REST API that is OpenAI API compatible, so you can use any OpenAI API compatible client, such as [Kubectl AI](https://github.com/sozercan/kubectl-ai), [Chatbot-UI](https://github.com/sozercan/chatbot-ui) and many more, to send requests to open LLMs!

- **[Fine-Tuning](https://kaito-project.github.io/aikit/docs/fine-tune)**: AIKit offers an extensible fine-tuning interface. It supports [Unsloth](https://github.com/unslothai/unsloth) for fast, memory efficient, and easy fine-tuning experience.

👉 For full documentation, please see [AIKit website](https://kaito-project.github.io/aikit/)!

## Features

- 🐳 No GPU, Internet access or additional tools needed except for [Docker](https://docs.docker.com/desktop/install/linux-install/)!
- 🤏 Minimal image size, resulting in less vulnerabilities and smaller attack surface with a custom [distroless](https://github.com/GoogleContainerTools/distroless)-based image
- 🎵 [Fine-tune support](https://kaito-project.github.io/aikit/docs/fine-tune)
- 🚀 Easy to use declarative configuration for [inference](https://kaito-project.github.io/aikit/docs/specs-inference) and [fine-tuning](https://kaito-project.github.io/aikit/docs/specs-finetune)
- ✨ OpenAI API compatible to use with any OpenAI API compatible client
- 📸 [Multi-modal model support](https://kaito-project.github.io/aikit/docs/vision)
- 🖼️ [Image generation support](https://kaito-project.github.io/aikit/docs/diffusion)
- 🦙 Support for GGUF ([`llama`](https://github.com/ggerganov/llama.cpp)), GPTQ or EXL2 ([`exllama2`](https://github.com/turboderp/exllamav2)), and GGML ([`llama-ggml`](https://github.com/ggerganov/llama.cpp)) models
- 🚢 [Kubernetes deployment ready](https://kaito-project.github.io/aikit/docs/kubernetes)
- 📦 Supports multiple models with a single image
- 🖥️ Supports [AMD64 and ARM64](https://kaito-project.github.io/aikit/docs/create-images#multi-platform-support) CPUs and [GPU-accelerated inferencing with NVIDIA GPUs](https://kaito-project.github.io/aikit/docs/gpu)
- 🔐 Ensure [supply chain security](https://kaito-project.github.io/aikit/docs/security) with SBOMs, Provenance attestations, and signed images
- 🌈 Supports air-gapped environments with self-hosted, local, or any remote container registries to store model images for inference on the edge.

## Quick Start

You can get started with AIKit quickly on your local machine without a GPU!

```bash
docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b
```

After running this, navigate to [http://localhost:8080/chat](http://localhost:8080/chat) to access the WebUI!

### API

AIKit provides an OpenAI API compatible endpoint, so you can use any OpenAI API compatible client to send requests to open LLMs!

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "explain kubernetes in a sentence"}]
  }'
```

Output should be similar to:

```jsonc
{
  // ...
    "model": "llama-3.1-8b-instruct",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Kubernetes is an open-source container orchestration system that automates the deployment, scaling, and management of applications and services, allowing developers to focus on writing code rather than managing infrastructure."
            }
        }
    ],
  // ...
}
```

That's it! 🎉 API is OpenAI compatible so this is a drop-in replacement for any OpenAI API compatible client.

## Pre-made Models

AIKit comes with pre-made models that you can use out-of-the-box!

If it doesn't include a specific model, you can always [create your own images](https://kaito-project.github.io/aikit/docs/create-images), and host in a container registry of your choice!

## CPU

> [!NOTE]
> AIKit supports both AMD64 and ARM64 CPUs. You can run the same command on either architecture, and Docker will automatically pull the correct image for your CPU.
>
> Depending on your CPU capabilities, AIKit will automatically select the most optimized instruction set.

| Model           | Optimization | Parameters | Command                                                                     | Model Name               | License                                                                            |
| --------------- | ------------ | ---------- | --------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| 🦙 Llama 3.2     | Instruct     | 1B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`   | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🦙 Llama 3.2     | Instruct     | 3B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`   | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🦙 Llama 3.1     | Instruct     | 8B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`   | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🦙 Llama 3.3     | Instruct     | 70B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b`  | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                        |  |
| Ⓜ️ Mixtral       | Instruct     | 8x7B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/mixtral:8x7b`  | `mixtral-8x7b-instruct`  | [Apache](https://choosealicense.com/licenses/apache-2.0/)                          |
| 🅿️ Phi 4         | Instruct     | 14B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`      | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)                  |
| 🔡 Gemma 2       | Instruct     | 2B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gemma2:2b`     | `gemma-2-2b-instruct`    | [Gemma](https://ai.google.dev/gemma/terms)                                         |
| QwQ             |              | 32B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwq:32b`       | `qwq-32b`        | [Apache 2.0](https://huggingface.co/Qwen/QwQ-32B/blob/main/LICENSE)        |
| ⌨️ Codestral 0.1 | Code         | 22B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/codestral:22b` | `codestral-22b`          | [MNLP](https://mistral.ai/licenses/MNPL-0.1.md)                                    |
| 🤖 GPT-OSS       |              | 20B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                      |
| 🤖 GPT-OSS       |              | 120B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                      |


### NVIDIA CUDA

> [!NOTE]
> To enable GPU acceleration, please see [GPU Acceleration](https://kaito-project.github.io/aikit/docs/gpu).
>
> Please note that only difference between CPU and GPU section is the `--gpus all` flag in the command to enable GPU acceleration.

| Model           | Optimization  | Parameters | Command                                                                                | Model Name               | License                                                                                                                     |
| --------------- | ------------- | ---------- | -------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 🦙 Llama 3.2     | Instruct      | 1B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`   | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.2     | Instruct      | 3B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`   | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.1     | Instruct      | 8B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`   | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.3     | Instruct      | 70B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b`  | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                                                                 |  |
| Ⓜ️ Mixtral       | Instruct      | 8x7B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/mixtral:8x7b`  | `mixtral-8x7b-instruct`  | [Apache](https://choosealicense.com/licenses/apache-2.0/)                                                                   |
| 🅿️ Phi 4         | Instruct      | 14B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`      | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)                                                          |
| 🔡 Gemma 2       | Instruct      | 2B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gemma2:2b`     | `gemma-2-2b-instruct`    | [Gemma](https://ai.google.dev/gemma/terms)                                                                                  |
| QwQ             |               | 32B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwq:32b`       | `qwq-32b`        | [Apache 2.0](https://huggingface.co/Qwen/QwQ-32B/blob/main/LICENSE)                                                 |
| ⌨️ Codestral 0.1 | Code          | 22B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/codestral:22b` | `codestral-22b`          | [MNLP](https://mistral.ai/licenses/MNPL-0.1.md)                                                                             |
| 📸 Flux 1 Dev    | Text to image | 12B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/flux1:dev`     | `flux-1-dev`             | [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| 🤖 GPT-OSS       |               | 20B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |
| 🤖 GPT-OSS       |               | 120B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |


### Apple Silicon (experimental)

> [!NOTE]
> To enable GPU acceleration on Apple Silicon, please see [Podman Desktop documentation](https://podman-desktop.io/docs/podman/gpu). For more information, please see [GPU Acceleration](https://kaito-project.github.io/aikit/docs/gpu).
>
> Apple Silicon is an _experimental_ runtime and it may change in the future. This runtime is specific to Apple Silicon only, and it will not work as expected on other architectures, including Intel Macs.
>
> Only `gguf` models are supported on Apple Silicon.

| Model       | Optimization | Parameters | Command                                                                                                  | Model Name              | License                                                                            |
| ----------- | ------------ | ---------- | -------------------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| 🦙 Llama 3.2 | Instruct     | 1B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:1b` | `llama-3.2-1b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🦙 Llama 3.2 | Instruct     | 3B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:3b` | `llama-3.2-3b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🦙 Llama 3.1 | Instruct     | 8B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.1:8b` | `llama-3.1-8b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                        |
| 🅿️ Phi 4     | Instruct     | 14B        | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/phi4:14b`    | `phi-4-14b-instruct`    | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)                  |
| 🔡 Gemma 2   | Instruct     | 2B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/gemma2:2b`   | `gemma-2-2b-instruct`   | [Gemma](https://ai.google.dev/gemma/terms)                                         |

## Contributing

Want to contribute to AIKit? Check out our [Contributing Guide](./CONTRIBUTING.md) for development setup, testing instructions, and contribution guidelines.

## What's next?

👉 For more information and how to fine tune models or create your own images, please see [AIKit website](https://kaito-project.github.io/aikit/)!
