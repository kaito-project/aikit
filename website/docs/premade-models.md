---
title: Pre-made models
---

Start with Qwen 3.5 4B for text chat. Use the smaller 2B preset when memory is limited, or choose a larger model for workloads that justify the extra memory and latency.

:::note
The new Qwen, Gemma 4, Devstral Small 2, and FLUX.2 presets are pending staging validation and image publication. The tables list their planned tags. GPT-OSS tags already exist, but the corrected recipes also need validation and republishing. Until publication, the maintained Llama and Phi images remain available. See the [model release process](./release.md#predefined-models).
:::

## Recommended presets

Prefix each image below with `ghcr.io/kaito-project/aikit/`. API model IDs are the same across runtimes. All recommended weights use Apache 2.0 licenses; the upstream model cards are linked below.

Gemma 4 uses Google's [Apache 2.0 license](https://ai.google.dev/gemma/apache_2).

| Model | Image | API model ID | Quantization | GGUF size |
| --- | --- | --- | --- | --- |
| [Qwen 3.5 2B](https://huggingface.co/Qwen/Qwen3.5-2B) | `qwen3.5:2b` | `qwen-3.5-2b` | Q4_K_M | 1.28 GB |
| [Qwen 3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B), quickstart | `qwen3.5:4b` | `qwen-3.5-4b` | Q4_K_M | 2.74 GB |
| [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `qwen3.5:9b` | `qwen-3.5-9b` | Q4_K_M | 5.68 GB |
| [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3.8:27b` | `qwen-3.8-27b` | UD-Q4_K_M | 16.46 GB |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | `gemma4:e2b` | `gemma-4-e2b-instruct` | QAT Q4_0 | 3.35 GB |
| [Devstral Small 2 24B](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512), coding | `devstral-small2:24b` | `devstral-small-2-24b-instruct` | Q4_K_M | 14.33 GB |
| [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) | `gpt-oss:20b` | `gpt-oss-20b` | MXFP4 | 12.11 GB |
| [GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b) | `gpt-oss:120b` | `gpt-oss-120b` | MXFP4 | 63.39 GB |
| [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), image generation | `flux2:klein-4b` | `flux-2-klein-4b` | BF16 | Downloads on first use |

Sizes are decimal GB of GGUF weights. Runtime memory must also cover the context cache and backend; container images include additional dependencies. Large models such as GPT-OSS 120B require substantially more memory and storage than the quickstart.

The GGUF presets serve text only. They do not include multimodal projectors, even when the upstream model supports image or audio inputs. Gemma's E2B name refers to effective parameters; the model has about 5.1B parameters including embeddings. See [Google's model card](https://huggingface.co/google/gemma-4-E2B-it).

New text presets and GPT-OSS use an 8192-token context by default. Qwen and Gemma default to `reasoning_effort: none`; a request can select another reasoning effort. GPT-OSS defaults to `medium`. When enabling Qwen reasoning, also follow the sampling recommendations in its upstream model card.

The GGUF downloads for these presets pin a Hugging Face revision and SHA-256 checksum. See the [recipes](https://github.com/kaito-project/aikit/tree/main/models) for exact artifact sources. FLUX.2 has different download behavior, described below.

## CPU and NVIDIA CUDA

Text images support AMD64 and ARM64. Docker selects the correct architecture, and LocalAI selects the CPU instruction set. The CUDA images include a digest-pinned CPU companion backend, so the same text image can run without an NVIDIA GPU.

```bash
docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b
```

For NVIDIA GPU acceleration, add `--gpus all` after configuring the [NVIDIA Container Toolkit](./gpu.md):

```bash
docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b
```

Use `qwen-3.5-4b` as the model in requests to `/v1/chat/completions`. Substitute another image and API model ID from the table to switch models.

## Image generation

FLUX.2 Klein 4B uses the `diffusers` backend on NVIDIA CUDA, Linux AMD64 only. The preset uses BF16, CPU offloading, four sampling steps, and guidance scale 1.0, following the [upstream example](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B#usage). Upstream reports about 13 GB VRAM with offloading; allow host memory and disk space for the full pipeline as well.

The recipe selects `runtime: cuda-12`, which uses the catalog's experimental LocalAI v4.10.0 Diffusers plan. It sets `runner: false` so LocalAI starts with the baked configuration and no model argument. The generic `cuda` runtime selects the older v3.12.1 plan. Validate the explicit CUDA 12 plan on the target GPU before publication.

```bash
docker run -d --rm --gpus all -p 8080:8080 ghcr.io/kaito-project/aikit/flux2:klein-4b
```

Send image-generation requests to `/v1/images/generations` with model `flux-2-klein-4b`. The image contains the backend and configuration. It downloads the pipeline from Hugging Face on first use, so it needs network access and a writable cache. Persist the Hugging Face cache to avoid downloading weights again after recreating the container. This preset does not pin the downloaded repository revision and is not ready for offline inference without preparing that cache.

## Maintained compatibility presets

These presets retain their current recipes, image tags, and API IDs for existing deployments. Full rebuilds refresh all published variants. Weekly patching currently covers canonical tags in the main image repository; aliases and Apple Silicon variants require a full rebuild through `update-models`.

| Model | Image | API model ID | License |
| --- | --- | --- | --- |
| Llama 3.2 1B | `llama3.2:1b` | `llama-3.2-1b-instruct` | [Llama](https://ai.meta.com/llama/license/) |
| Llama 3.2 3B | `llama3.2:3b` | `llama-3.2-3b-instruct` | [Llama](https://ai.meta.com/llama/license/) |
| Llama 3.1 8B | `llama3.1:8b` | `llama-3.1-8b-instruct` | [Llama](https://ai.meta.com/llama/license/) |
| Llama 3.3 70B | `llama3.3:70b` | `llama-3.3-70b-instruct` | [Llama](https://ai.meta.com/llama/license/) |
| Phi 4 14B | `phi4:14b` | `phi-4-14b-instruct` | [MIT](https://huggingface.co/microsoft/phi-4/blob/main/LICENSE) |

The existing `-instruct` tag aliases remain available. These image names also use the prefix `ghcr.io/kaito-project/aikit/`.

## Apple Silicon, experimental

The publishing workflow includes Apple Silicon variants of the GGUF text presets under `ghcr.io/kaito-project/aikit/applesilicon/`. FLUX.2 is excluded. This experimental runtime targets Apple Silicon, not Intel Macs. Model support and available memory still need validation on the target device.

Set up GPU access using the [Podman Desktop documentation](https://podman-desktop.io/docs/podman/gpu), then use:

```bash
podman run -d --rm --device /dev/dri -p 8080:8080 ghcr.io/kaito-project/aikit/applesilicon/qwen3.5:4b
```

## AMD ROCm, experimental

Pre-made ROCm images are not published. For AMD GPUs, [create a custom image](./create-images.md) using `llama-cpp` with `runtime: rocm`, then follow the ROCm device setup in [GPU acceleration](./gpu.md).

The documented ROCm path uses `llama-cpp` on `linux/amd64`. Other experimental tuples may appear in a frontend release's backend catalog; check that release's lock before using one.

## Deprecated models

The following presets are removed from the active catalog and receive no further image rebuilds or weekly patches. Existing registry images are retained. Migrate by changing both the image tag and the API model ID, then validate behavior on your workload.

| Retired preset | Historical image | Recommended replacement |
| --- | --- | --- |
| Gemma 2 2B | `ghcr.io/kaito-project/aikit/gemma2:2b` | `gemma4:e2b`, API `gemma-4-e2b-instruct` |
| Mixtral 8x7B | `ghcr.io/kaito-project/aikit/mixtral:8x7b` | `qwen3.8:27b`, API `qwen-3.8-27b` |
| QwQ 32B | `ghcr.io/kaito-project/aikit/qwq:32b` | `qwen3.8:27b`, API `qwen-3.8-27b`; enable reasoning as needed |
| Codestral 22B | `ghcr.io/kaito-project/aikit/codestral:22b` | `devstral-small2:24b`, API `devstral-small-2-24b-instruct` |
| FLUX.1 Dev | `ghcr.io/kaito-project/aikit/flux1:dev` | `flux2:klein-4b`, API `flux-2-klein-4b` |

Devstral Small 2 replaces Codestral for coding assistants. It is not a drop-in replacement for Codestral's fill-in-the-middle prompting. FLUX.2 changes the generation pipeline and sampling defaults. Validate these migrations before switching production traffic.

### Previously deprecated models

The older image references below are retained for existing users. They are no longer updated. You can [create your own images](./create-images.md) if you need to maintain these models.

#### CPU

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


#### NVIDIA CUDA

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
