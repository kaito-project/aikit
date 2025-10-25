---
title: Fine Tuning
---

Fine tuning process allows the adaptation of pre-trained models to domain-specific data. At this time, AIKit fine tuning process is only supported with NVIDIA GPUs.

:::note
Due to limitations with BuildKit and NVIDIA, it is essential that the GPU driver version on your host matches the version AIKit will install in the container during the build process.

To determine your host GPU driver version, you can execute `nvidia-smi` or `cat /proc/driver/nvidia/version`.

For information on the GPU driver versions supported by AIKit, please visit https://download.nvidia.com/XFree86/Linux-x86_64/.

Should your host GPU driver version not be listed, you will need to update to a compatible version available in the NVIDIA downloads mentioned above. It's important to note that there's no need to directly install drivers from the NVIDIA downloads; the versions simply need to be consistent.

We hope to optimize this process in the future to eliminate this requirement.
:::

## Getting Started

To get started, you need to create a builder to be able to access host GPU devices.

Create a builder with the following configuration:

```bash
docker buildx create --name aikit-builder --use --buildkitd-flags '--allow-insecure-entitlement security.insecure'
```

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
docker buildx build --builder aikit-builder --allow security.insecure --file "/path/to/config.yaml" --output "/path/to/output" --target unsloth --progress plain .
```

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

### Build fails with `ERROR 404: Not Found.`

This issue arises from a discrepancy between the GPU driver versions on your host and the container. Unfortunately, a matching version for your host driver is not available in the NVIDIA downloads at this time. For further details, please consult the note provided at the beginning of this page.

If you are on Windows Subsystem for Linux (WSL), WSL doesn't expose the host driver version information on `/proc/driver/nvidia/version`. Due to this limitation, WSL is not supported at this time.
