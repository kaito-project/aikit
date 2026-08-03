---
title: Fine Tuning
---

Fine tuning process allows the adaptation of pre-trained models to domain-specific data. At this time, AIKit fine tuning process is only supported with NVIDIA GPUs; AMD ROCm is not supported yet.

:::note
Fine tuning requires Docker Engine 27 or later, Buildx 0.22 or later, BuildKit 0.20 or later, and NVIDIA Container Toolkit configured with CDI support. AIKit requests the first GPU as `nvidia.com/gpu=0` for its training and export build steps instead of creating NVIDIA device nodes or installing a matching driver inside the build container.

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

The GPU-capable BuildKit image is experimental. The inspection output must list `nvidia.com/gpu=0` or the on-demand NVIDIA device kind `nvidia.com/gpu` under `Devices`. If it does not, ensure the NVIDIA CDI specification and hook are available to the BuildKit daemon before continuing. The standard CDI directories are `/etc/cdi`, `/var/run/cdi`, and `/etc/buildkit/cdi`.

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
  - source: "yahma/alpaca-cleaned" # data set to be used for fine tuning. This can be a Huggingface dataset or a URL pointing to a JSON file
    type: "alpaca" # type of dataset. only alpaca is supported at this time.
config:
  unsloth:
```

For full configuration, please refer to [Fine Tune API Specifications](./specs-finetune.md).

:::note
Please refer to [Unsloth documentation](https://github.com/unslothai/unsloth) for more information about Unsloth configuration.
:::

#### Example Configuration

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the example below.
:::

https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth.yaml


## Build

Build using following command and make sure to replace `--target` with the fine-tuning implementation of your choice (`unsloth` is the only option supported at this time), `--file` with the path to your configuration YAML and `--output` with the output directory of the finetuned model.

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

### Build fails because `nvidia.com/gpu=0` is not registered

Run `nvidia-ctk cdi list` on the host and `docker buildx inspect aikit-builder --bootstrap`. The device must be present in both outputs. If it is missing from the builder, make the NVIDIA CDI specification and hook available to the BuildKit daemon and recreate or restart the builder.

### Build fails with `requested by the build but not allowed`

Enable the `device` entitlement on the BuildKit daemon with `--allow-insecure-entitlement device`, and pass `--allow 'device=nvidia.com/gpu=0'` to the build command.
