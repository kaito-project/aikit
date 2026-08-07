---
title: Fine Tuning
---

Fine tuning process allows the adaptation of pre-trained models to domain-specific data. At this time, AIKit fine tuning process is only supported with NVIDIA GPUs; AMD ROCm is not supported yet.

:::note
Fine tuning requires Docker Engine 27 or later, Buildx 0.22 or later, BuildKit 0.20 or later, and NVIDIA Container Toolkit configured with CDI support. AIKit requests the first GPU as `nvidia.com/gpu=0` by default for its training and export build steps instead of creating NVIDIA device nodes or installing a matching driver inside the build container. Supply an immutable `GPU-...` or `MIG-...` selector through `cdiDevice` when cross-session GPU result-cache reuse is required.

On Windows Subsystem for Linux, use Docker Desktop with WSL2 GPU paravirtualization and Buildx 0.27 or later. Earlier Buildx versions do not expose the WSL driver libraries to a containerized GPU builder. See Docker's [CDI build documentation](https://docs.docker.com/build/building/cdi/).

Verify the host prerequisites before creating the builder:

```bash
nvidia-smi
nvidia-ctk cdi list
docker version
docker buildx version
```
:::

## Getting Started

To get started, you need to create a builder to be able to access host GPU devices.

Create a builder with the following configuration:

```bash
docker buildx create --name aikit-builder --use \
  --driver docker-container \
  --driver-opt image=moby/buildkit:buildx-stable-1-gpu \
  --buildkitd-flags '--allow-insecure-entitlement device'
docker buildx inspect aikit-builder --bootstrap
```

The GPU-capable BuildKit image is experimental. The inspection output must list a usable NVIDIA selector such as `nvidia.com/gpu=0`, an immutable UUID selector, or the on-demand NVIDIA device kind `nvidia.com/gpu` under `Devices`. If it does not, ensure the NVIDIA CDI specification and hook are available to the BuildKit daemon before continuing. The standard CDI directories are `/etc/cdi`, `/var/run/cdi`, and `/etc/buildkit/cdi`.

:::tip
Additionally, you can build using other BuildKit drivers, such as [Kubernetes driver](https://docs.docker.com/build/drivers/kubernetes/) by setting `--driver=kubernetes` if you are interested in building using a Kubernetes cluster. Please see [BuildKit Drivers](https://docs.docker.com/build/drivers/) for more information.
:::

## Targets and Configuration

AIKit is capable of supporting multiple fine tuning implementation targets. At this time, [Unsloth](https://github.com/unslothai/unsloth) is the only supported target, but can be extended for other fine tuning implementations in the future.

### Unsloth

Create a YAML file with your configuration. For example, minimum config looks like:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
baseModel: "unsloth/llama-2-7b-bnb-4bit" # base model to be fine tuned. this can be any model from Huggingface. For unsloth optimized base models, see https://huggingface.co/unsloth
datasets:
  - source: "yahma/alpaca-cleaned" # data set to be used for fine tuning. This can be a Huggingface dataset or a URL pointing to a JSON or JSON Lines file
    type: "alpaca" # supported types are alpaca, prompt-completion, and text
config:
  unsloth:
```

For full configuration, please refer to [Fine Tune API Specifications](./specs-finetune.md).

#### Dataset Types

AIKit supports one dataset per fine-tuning configuration. The configured `type` selects the record schema and training loss behavior; unknown types fail instead of falling back to another formatter.

##### Alpaca

The `alpaca` type uses the existing `instruction`, `input`, and `output` string columns. AIKit renders the existing Alpaca prompt, appends EOS explicitly, and trains with full-sequence loss. Existing Alpaca configurations require no changes.

##### Prompt-Completion

The `prompt-completion` type keeps the `prompt` and `completion` columns separate. Every record must contain a string `prompt` and a non-empty string `completion`. Prompt tokens are masked from the loss, while completion and EOS tokens are supervised. Empty datasets and missing, null, or invalid fields are rejected before model allocation.

```yaml
datasets:
  - source: organization/question-answer-data
    type: prompt-completion
```

An expected JSON Lines record is:

```json
{"prompt":"Question: What is a container image?\nAnswer:","completion":" An immutable package containing an application and its dependencies."}
```

##### Text

The `text` type accepts complete, preformatted training sequences in a non-empty string `text` column. AIKit preserves the sequence content while normalizing its special-token boundaries: tokenization produces exactly one effective leading BOS where applicable and every record ends with exactly one EOS. Tokenizers that add BOS or EOS automatically, as well as tokenizers that do not define BOS, are supported, but the tokenizer must define a usable EOS token. Records whose normalized token sequence exceeds `maxSeqLength` are rejected rather than truncated so that their terminal EOS is retained. All tokens in the normalized record are supervised, and packing retains the tokenized EOS record boundaries.

```yaml
datasets:
  - source: organization/domain-corpus
    type: text
```

An expected JSON Lines record is:

```json
{"text":"Question: What is a container image?\nAnswer: An immutable package containing an application and its dependencies."}
```

:::caution
The `text` type performs full-sequence supervised fine-tuning (SFT). It is not continued pretraining: AIKit retains the standard LoRA targets and optimizer configuration and does not train embedding or language-model-head parameters.
:::

:::note
Please refer to [Unsloth documentation](https://github.com/unslothai/unsloth) for more information about Unsloth configuration.
:::

#### Example Configurations

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the example below.
:::

- [Alpaca](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth.yaml)
- [Prompt-completion smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-prompt-completion-smoke.yaml)
- [Text smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-text-smoke.yaml)


## Build

Build using following command and make sure to replace `--target` with the fine-tuning implementation of your choice (`unsloth` is the only option supported at this time), `--file` with the path to your configuration YAML and `--output` with the output directory of the finetuned model. This example assumes the builder and Docker runtime use the same local NVIDIA device namespace. For a remote or on-demand builder, select a device shown by `docker buildx inspect` and omit `nvidiaDriverVersion` unless its driver version is known.

```bash
NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
NVIDIA_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)
NVIDIA_CDI_DEVICE="nvidia.com/gpu=${NVIDIA_GPU_UUID}"

docker buildx build --builder aikit-builder \
  --allow "device=${NVIDIA_CDI_DEVICE}" \
  --build-arg "nvidiaDriverVersion=${NVIDIA_DRIVER_VERSION}" \
  --build-arg "cdiDevice=${NVIDIA_CDI_DEVICE}" \
  --file "/path/to/config.yaml" \
  --output "/path/to/output" \
  --target unsloth \
  --progress plain .
```

The `device` entitlement authorizes only the CDI device requested by AIKit. Training and GGUF export run with the normal BuildKit sandbox security mode. The `cdiDevice` argument selects that GPU, while the optional `nvidiaDriverVersion` argument is used only as a GPU-phase cache discriminator; AIKit does not install that driver in the build container. AIKit reuses GPU results across BuildKit sessions only when the driver version is paired with an immutable `GPU-...` or `MIG-...` CDI selector. Index, `all`, and on-demand selectors can identify different hardware on another builder, so AIKit falls back to the current BuildKit session as the result-cache discriminator while retaining persistent model, dataset, and compiler caches.

The training and GGUF export phases are cached separately. Changing only `output.name` reuses both phases, while changing only `output.quantize` reuses training and reruns export.

Depending on your setup and configuration, build process may take some time. At the end of the build, the fine-tuned model will automatically be quantized with the specified format and output to the path specified in the `--output`.

Output will be a `GGUF` model file with the name and quanization format from the configuration. For example:

```bash
$ ls -al _output
-rw-r--r--  1 kaito-project kaito-project 7161089856 Mar  3 00:19 aikit-model-q4_k_m.gguf
```

## Demo

https://www.youtube.com/watch?v=FZuVb-9i-94

## What's next?

👉 Now that you have a fine-tuned model output as a GGUF file, you can refer to [Creating Model Images](./create-images.md) on how to create an image with AIKit to serve your fine-tuned model!

## Troubleshooting

### Build fails with `failed to solve: DeadlineExceeded: context deadline exceeded`

This is a known issue with BuildKit and might be related to disk speed. For more information, please see https://github.com/moby/buildkit/issues/4327

### Build fails because the requested NVIDIA CDI device is not registered

Run `nvidia-ctk cdi list` on the host and `docker buildx inspect aikit-builder --bootstrap`. The requested selector must be available in the environment where it is used. A local builder can use the same UUID listed by the host, while an on-demand builder may initially expose only `nvidia.com/gpu`. If no usable selector appears in the builder, make the NVIDIA CDI specification and hook available to the BuildKit daemon and recreate or restart the builder.

### Build fails with `requested by the build but not allowed`

Enable the `device` entitlement on the BuildKit daemon with `--allow-insecure-entitlement device`, and pass `--allow "device=${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=0}"` to the build command.
