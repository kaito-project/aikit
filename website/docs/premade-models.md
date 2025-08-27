---
title: Model Gallery
---

AIKit comes with a model gallery that you can use out-of-the-box!

If it doesn't include a specific model, you can always [create your own images](https://kaito-project.github.io/aikit/premade-models/), and host in a container registry of your choice!

## CPU

:::note
AIKit supports both AMD64 and ARM64 CPUs. You can run the same command on either architecture, and Docker will automatically pull the correct image for your CPU.
Depending on your CPU capabilities, AIKit will automatically select the most optimized instruction set.
:::

| Model           | Optimization | Parameters | Command                                                                     | Model Name               | License                                                             |
| --------------- | ------------ | ---------- | --------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------- |
| 🦙 Llama 3.2     | Instruct     | 1B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`   | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.2     | Instruct     | 3B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`   | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.1     | Instruct     | 8B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`   | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.3     | Instruct     | 70B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b`  | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                         |  |
| Ⓜ️ Mixtral       | Instruct     | 8x7B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/mixtral:8x7b`  | `mixtral-8x7b-instruct`  | [Apache](https://choosealicense.com/licenses/apache-2.0/)           |
| 🅿️ Phi 4         | Instruct     | 14B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`      | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)  |
| 🔡 Gemma 3       | Instruct     | 4B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gemma3:4b`     | `gemma-3-4b-instruct`    | [Gemma](https://ai.google.dev/gemma/terms)                          |
| ⌨️ Codestral 0.1 | Code         | 22B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/codestral:22b` | `codestral-22b`          | [MNLP](https://mistral.ai/licenses/MNPL-0.1.md)                     |
| 🥺 QwQ           |              | 32B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwq:32b`       | `qwq-32b`                | [Apache 2.0](https://huggingface.co/Qwen/QwQ-32B/blob/main/LICENSE) |  |
| 🤖 GPT-OSS       |              | 20B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)       |
| 🤖 GPT-OSS       |              | 120B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)       |

## NVIDIA CUDA

| Model           | Optimization  | Parameters | Command                                                                                | Model Name               | License                                                                                                                     |
| --------------- | ------------- | ---------- | -------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 🦙 Llama 3.2     | Instruct      | 1B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`   | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.2     | Instruct      | 3B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`   | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.1     | Instruct      | 8B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`   | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                                                                                 |
| 🦙 Llama 3.3     | Instruct      | 70B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b`  | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                                                                                 |  |
| Ⓜ️ Mixtral       | Instruct      | 8x7B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/mixtral:8x7b`  | `mixtral-8x7b-instruct`  | [Apache](https://choosealicense.com/licenses/apache-2.0/)                                                                   |
| 🅿️ Phi 4         | Instruct      | 14B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`      | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)                                                          |
| 🔡 Gemma 3       | Instruct      | 4B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gemma3:4b`     | `gemma-3-4b-instruct`    | [Gemma](https://ai.google.dev/gemma/terms)                                                                                  |
| ⌨️ Codestral 0.1 | Code          | 22B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/codestral:22b` | `codestral-22b`          | [MNLP](https://mistral.ai/licenses/MNPL-0.1.md)                                                                             |
| 🥺 QwQ           |               | 32B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwq:32b`       | `qwq-32b`                | [Apache 2.0](https://huggingface.co/Qwen/QwQ-32B/blob/main/LICENSE)                                                         |
| 📸 Flux 1 Dev    | Text to image | 12B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/flux1:dev`     | `flux-1-dev`             | [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| 🤖 GPT-OSS       |               | 20B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |
| 🤖 GPT-OSS       |               | 120B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |

:::note
Please see [models folder](https://github.com/kaito-project/aikit/tree/main/models) for model gallery definitions.

If not being offloaded to GPU VRAM, minimum of 8GB of RAM is required for 7B models, 16GB of RAM to run 13B models, and 32GB of RAM to run 8x7B models.

All model gallery images include CUDA v12 libraries. They are used with [NVIDIA GPU acceleration](gpu.md). If a supported NVIDIA GPU is not found in your system, AIKit will automatically fallback to CPU with the most optimized runtime (`avx2`, `avx`, or `fallback`).
:::

## Apple Silicon (experimental)

:::note
To enable GPU acceleration on Apple Silicon, please see [Podman Desktop documentation](https://podman-desktop.io/docs/podman/gpu).

Apple Silicon is an _experimental_ runtime and it may change in the future. This runtime is specific to Apple Silicon only, and it will not work as expected on other architectures, including Intel Macs.

Only `gguf` models are supported on Apple Silicon.
:::

| Model       | Optimization | Parameters | Command                                                                                                  | Model Name              | License                                                            |
| ----------- | ------------ | ---------- | -------------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------ |
| 🦙 Llama 3.2 | Instruct     | 1B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:1b` | `llama-3.2-1b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.2 | Instruct     | 3B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:3b` | `llama-3.2-3b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.1 | Instruct     | 8B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.1:8b` | `llama-3.1-8b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🅿️ Phi 4     | Instruct     | 14B        | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/phi4:14b`    | `phi-4-14b-instruct`    | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE) |
| 🔡 Gemma 2   | Instruct     | 2B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/gemma2:2b`   | `gemma-2-2b-instruct`   | [Gemma](https://ai.google.dev/gemma/terms)                         |

## Deprecated Models

The following pre-made models are deprecated and no longer updated. Images will continue to be pullable, if needed.

If you need to use these specific models, you can always [create your own images](./create-images.md), and host in a container registry of your choice!

### CPU

| Model       | Optimization | Parameters | Command                                                                   | License                                                                             |
| ----------- | ------------ | ---------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 🐬 Orca 2    |              | 13B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/aikit/orca2:13b`        | [Microsoft Research](https://huggingface.co/microsoft/Orca-2-13b/blob/main/LICENSE) |
| 🅿️ Phi 2     | Instruct     | 2.7B       | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/phi2:2.7b`              | [MIT](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)                  |
| 🅿️ Phi 3     | Instruct     | 3.8B       | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/phi3:3.8b`              | `phi-3-3.8b`                                                                        | [MIT](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE) |
| 🅿️ Phi 3.5   | Instruct     | 3.8B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/phi3.5:3.8b` | [MIT](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE)  |
| 🦙 Llama 3   | Instruct     | 8B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama3:8b`              | `llama-3-8b-instruct`                                                               | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 3   | Instruct     | 70B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama3:70b`             | `llama-3-70b-instruct`                                                              | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 7B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama2:7b`              | `llama-2-7b-chat`                                                                   | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 13B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama2:13b`             | `llama-2-13b-chat`                                                                  | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🔡 Gemma 1.1 | Instruct     | 2B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/gemma:2b`               | `gemma-2b-instruct`                                                                 | [Gemma](https://ai.google.dev/gemma/terms)                                          |
| 🔡 Gemma 2   | Instruct     | 2B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gemma2:2b`   | `gemma-2-2b-instruct`                                                               | [Gemma](https://ai.google.dev/gemma/terms)                                          |


### NVIDIA CUDA

| Model       | Optimization | Parameters | Command                                                                              | License                                                                             |
| ----------- | ------------ | ---------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| 🐬 Orca 2    |              | 13B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/orca2:13b-cuda`         | [Microsoft Research](https://huggingface.co/microsoft/Orca-2-13b/blob/main/LICENSE) |
| 🅿️ Phi 2     | Instruct     | 2.7B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/phi2:2.7b-cuda`         | [MIT](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)                  |
| 🅿️ Phi 3     | Instruct     | 3.8B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/phi3:3.8b`              | `phi-3-3.8b`                                                                        | [MIT](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE) |
| 🅿️ Phi 3.5   | Instruct     | 3.8B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/phi3.5:3.8b` | [MIT](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE)  |
| 🦙 Llama 3   | Instruct     | 8B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama3:8b`              | `llama-3-8b-instruct`                                                               | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 3   | Instruct     | 70B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama3:70b`             | `llama-3-70b-instruct`                                                              | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 7B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama2:7b`              | `llama-2-7b-chat`                                                                   | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 13B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama2:13b`             | `llama-2-13b-chat`                                                                  | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🔡 Gemma 1.1 | Instruct     | 2B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/gemma:2b`               | `gemma-2b-instruct`                                                                 | [Gemma](https://ai.google.dev/gemma/terms)                                          |
| 🔡 Gemma 2   | Instruct     | 2B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gemma2:2b`   | `gemma-2-2b-instruct`                                                               | [Gemma](https://ai.google.dev/gemma/terms)                                          |
