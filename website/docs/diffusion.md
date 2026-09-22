---
title: Diffusion
---

AIKit documents the [`diffusers`](#diffusers) catalog family for image generation.

## diffusers

The `diffusers` backend uses the Hugging Face [`diffusers`](https://huggingface.co/docs/diffusers/en/index) library to generate images. The currently supported default and runner-enabled plan targets NVIDIA CUDA on Linux AMD64. A frontend release can also contain experimental `diffusers` plans for other runtimes or platforms in standard mode; catalog presence means AIKit can materialize that install plan, not that every model and device combination has been validated end to end. See [Backend catalog selection](specs-inference.md#backend-catalog-selection).

### Example

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the examples below.
:::

https://github.com/kaito-project/aikit/blob/main/test/aikitfile-diffusers.yaml

## stablediffusion NCNN

https://github.com/EdVince/Stable-Diffusion-NCNN

This backend:
- provides support for Stable Diffusion models
- does not support CUDA runtime yet

:::note
This has been deprecated as of `v0.18.0` release.
:::

### Example

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the examples below.
:::

https://github.com/kaito-project/aikit/blob/main/test/aikitfile-stablediffusion.yaml

### Demo

https://www.youtube.com/watch?v=gh7b-rt70Ug
