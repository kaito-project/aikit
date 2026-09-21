---
title: Pre-made Models
---

AIKit comes with pre-made models that you can use out-of-the-box!

If it doesn't include a specific model, you can always [create your own images](./create-images.md), and host in a container registry of your choice!

## CPU

:::note
AIKit supports both AMD64 and ARM64 CPUs. You can run the same command on either architecture, and Docker will automatically pull the correct image for your CPU.
Depending on your CPU capabilities, AIKit will automatically select the most optimized instruction set.
The GGUF chat presets listed below serve text only, even when the upstream model supports image or audio inputs.
:::

| Model           | Optimization | Parameters | Command                                                                     | Model Name               | License                                                             |
| --------------- | ------------ | ---------- | --------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------- |
| Qwen 3.5 | Instruct | 2B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:2b` | `qwen-3.5-2b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 4B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b` | `qwen-3.5-4b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 9B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:9b` | `qwen-3.5-9b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.8 | Instruct | 27B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.8:27b` | `qwen-3.8-27b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🔡 Gemma 4 E2B | Instruct | 5.1B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gemma4:e2b` | `gemma-4-e2b-instruct` | [Apache 2.0](https://ai.google.dev/gemma/apache_2) |
| Devstral Small 2 | Code | 24B | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/devstral-small2:24b` | `devstral-small-2-24b-instruct` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🦙 Llama 3.2     | Instruct     | 1B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`   | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.2     | Instruct     | 3B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`   | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.1     | Instruct     | 8B         | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`   | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                         |
| 🦙 Llama 3.3     | Instruct     | 70B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b`  | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                         |  |
| 🅿️ Phi 4         | Instruct     | 14B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`      | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE)  |
| 🤖 GPT-OSS       |              | 20B        | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)       |
| 🤖 GPT-OSS       |              | 120B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)       |

## NVIDIA CUDA

| Model       | Optimization | Parameters | Command                                                                               | Model Name               | License                                                            |
| ----------- | ------------ | ---------- | ------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| Qwen 3.5 | Instruct | 2B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:2b` | `qwen-3.5-2b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 4B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b` | `qwen-3.5-4b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 9B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:9b` | `qwen-3.5-9b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.8 | Instruct | 27B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.8:27b` | `qwen-3.8-27b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🔡 Gemma 4 E2B | Instruct | 5.1B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gemma4:e2b` | `gemma-4-e2b-instruct` | [Apache 2.0](https://ai.google.dev/gemma/apache_2) |
| Devstral Small 2 | Code | 24B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/devstral-small2:24b` | `devstral-small-2-24b-instruct` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 📸 Flux 2 Klein | Text to image | 4B | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/flux2:klein-4b` | `flux-2-klein-4b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🦙 Llama 3.2 | Instruct     | 1B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:1b`  | `llama-3.2-1b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.2 | Instruct     | 3B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.2:3b`  | `llama-3.2-3b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.1 | Instruct     | 8B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.1:8b`  | `llama-3.1-8b-instruct`  | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.3 | Instruct     | 70B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/llama3.3:70b` | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |  |
| 🅿️ Phi 4     | Instruct     | 14B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/phi4:14b`     | `phi-4-14b-instruct`     | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE) |
| 🤖 GPT-OSS       |               | 20B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:20b`   | `gpt-oss-20b`            | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |
| 🤖 GPT-OSS       |               | 120B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/gpt-oss:120b`  | `gpt-oss-120b`           | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                               |

:::note
Please see [models folder](https://github.com/kaito-project/aikit/tree/main/models) for pre-made model definitions.

If not being offloaded to GPU VRAM, minimum of 8GB of RAM is required for 7B models, 16GB of RAM to run 13B models, and 32GB of RAM to run 8x7B models.

The published `llama-cpp` model images in the NVIDIA CUDA section above use a CUDA 12 catalog plan that deliberately includes a digest-pinned CPU companion backend. When no compatible NVIDIA GPU is available, LocalAI can use that installed CPU backend. This is runtime behavior inside the selected CUDA plan, not catalog resolution silently changing to a CPU tuple.

FLUX.2 Klein 4B requires NVIDIA CUDA on Linux AMD64. It uses BF16 with CPU offloading; [upstream reports about 13 GB VRAM](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B#usage). Allow host memory and disk space for the full pipeline as well.

FLUX.2 downloads its pipeline on first use and needs network access and a writable Hugging Face cache. Persist the cache to avoid downloading the weights again after recreating the container. The download is not revision-pinned.
:::

## AMD ROCm (experimental)

:::note
Published pre-made model images are currently CUDA-based, so ROCm-accelerated images are not published yet.

To use AMD GPUs, create your own `llama-cpp` image with `runtime: rocm`, then run it with the ROCm device flags described in [GPU Acceleration](gpu.md). See [Creating Model Images](./create-images.md) for build examples.

The documented ROCm image path uses `llama-cpp` on `linux/amd64`. Other experimental ROCm tuples can appear in a frontend's catalog for standard builds; consult that release's lock before relying on one.
:::

## Apple Silicon (experimental)

:::note
To enable GPU acceleration on Apple Silicon, please see [Podman Desktop documentation](https://podman-desktop.io/docs/podman/gpu).

Apple Silicon is an _experimental_ runtime and it may change in the future. This runtime is specific to Apple Silicon only, and it will not work as expected on other architectures, including Intel Macs.

The published Apple Silicon images use the experimental `llama-cpp` Apple Silicon profile with GGUF models. Other experimental catalog tuples, if present, are not a promise that a published image or end-to-end model workflow is available.
:::

| Model       | Optimization | Parameters | Command                                                                                                  | Model Name              | License                                                            |
| ----------- | ------------ | ---------- | -------------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------ |
| Qwen 3.5 | Instruct | 2B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.5:2b` | `qwen-3.5-2b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 4B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.5:4b` | `qwen-3.5-4b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.5 | Instruct | 9B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.5:9b` | `qwen-3.5-9b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| Qwen 3.8 | Instruct | 27B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.8:27b` | `qwen-3.8-27b` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🔡 Gemma 4 E2B | Instruct | 5.1B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/gemma4:e2b` | `gemma-4-e2b-instruct` | [Apache 2.0](https://ai.google.dev/gemma/apache_2) |
| Devstral Small 2 | Code | 24B | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/devstral-small2:24b` | `devstral-small-2-24b-instruct` | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| 🦙 Llama 3.2 | Instruct     | 1B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:1b` | `llama-3.2-1b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.2 | Instruct     | 3B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.2:3b` | `llama-3.2-3b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🦙 Llama 3.1 | Instruct     | 8B         | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/llama3.1:8b` | `llama-3.1-8b-instruct` | [Llama](https://ai.meta.com/llama/license/)                        |
| 🅿️ Phi 4     | Instruct     | 14B        | `podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/phi4:14b`    | `phi-4-14b-instruct`    | [MIT](https://huggingface.co/microsoft/Phi-4/resolve/main/LICENSE) |

## Deprecated Models

The following pre-made models are deprecated and no longer updated. Images will continue to be pullable, if needed.

If you need to use these specific models, you can always [create your own images](./create-images.md), and host in a container registry of your choice!

| Retired preset | Historical image | Recommended replacement |
| --- | --- | --- |
| Gemma 2 2B | `ghcr.io/kaito-project/aikit/gemma2:2b` | `gemma4:e2b`, API `gemma-4-e2b-instruct` |
| Mixtral 8x7B | `ghcr.io/kaito-project/aikit/mixtral:8x7b` | `qwen3.8:27b`, API `qwen-3.8-27b` |
| QwQ 32B | `ghcr.io/kaito-project/aikit/qwq:32b` | `qwen3.8:27b`, API `qwen-3.8-27b`; enable reasoning as needed |
| Codestral 22B | `ghcr.io/kaito-project/aikit/codestral:22b` | `devstral-small2:24b`, API `devstral-small-2-24b-instruct` |
| FLUX.1 Dev | `ghcr.io/kaito-project/aikit/flux1:dev` | `flux2:klein-4b`, API `flux-2-klein-4b` |

Change both the image tag and API model name when migrating. Devstral Small 2 is not a drop-in replacement for Codestral's fill-in-the-middle prompting. FLUX.2 changes the generation pipeline and sampling defaults.

### CPU

| Model       | Optimization | Parameters | Command                                                            | License                                                                             |
| ----------- | ------------ | ---------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| 🐬 Orca 2    |              | 13B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/aikit/orca2:13b` | [Microsoft Research](https://huggingface.co/microsoft/Orca-2-13b/blob/main/LICENSE) |
| 🅿️ Phi 2     | Instruct     | 2.7B       | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/phi2:2.7b`       | [MIT](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)                  |
| 🅿️ Phi 3     | Instruct     | 3.8B       | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/phi3:3.8b`       | `phi-3-3.8b`                                                                        | [MIT](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE) |
| 🅿️ Phi 3.5   | Instruct     | 3.8B       | `docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/phi3.5:3.8b`    | [MIT](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE) |
| 🦙 Llama 3   | Instruct     | 8B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama3:8b`       | `llama-3-8b-instruct`                                                               | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 3   | Instruct     | 70B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama3:70b`      | `llama-3-70b-instruct`                                                              | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 7B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama2:7b`       | `llama-2-7b-chat`                                                                   | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 13B        | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/llama2:13b`      | `llama-2-13b-chat`                                                                  | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🔡 Gemma 1.1 | Instruct     | 2B         | `docker run -d --rm -p 8080:8080 ghcr.io/sozercan/gemma:2b`        | `gemma-2b-instruct`                                                                 | [Gemma](https://ai.google.dev/gemma/terms)                                          |


### NVIDIA CUDA

| Model | Optimization | Parameters | Command | License |
| ----- | ------------ | ---------- | ------- | ------- |
| 🐬 Orca 2    |              | 13B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/orca2:13b-cuda` | [Microsoft Research](https://huggingface.co/microsoft/Orca-2-13b/blob/main/LICENSE) |
| 🅿️ Phi 2     | Instruct     | 2.7B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/phi2:2.7b-cuda` | [MIT](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)                  |
| 🅿️ Phi 3     | Instruct     | 3.8B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/phi3:3.8b`      | `phi-3-3.8b`                                                                        | [MIT](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE) |
| 🅿️ Phi 3.5   | Instruct     | 3.8B       | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/phi3.5:3.8b`    | [MIT](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE) |
| 🦙 Llama 3   | Instruct     | 8B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama3:8b`      | `llama-3-8b-instruct`                                                               | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 3   | Instruct     | 70B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama3:70b`     | `llama-3-70b-instruct`                                                              | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 7B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama2:7b`      | `llama-2-7b-chat`                                                                   | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🦙 Llama 2   | Chat         | 13B        | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/llama2:13b`     | `llama-2-13b-chat`                                                                  | [Llama](https://ai.meta.com/llama/license/)                                         |
| 🔡 Gemma 1.1 | Instruct     | 2B         | `docker run -d --rm --gpus all -p 8080:8080 ghcr.io/sozercan/gemma:2b`       | `gemma-2b-instruct`                                                                 | [Gemma](https://ai.google.dev/gemma/terms)                                          |
