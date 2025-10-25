---
title: Diffusion
---

AIKit supports [`diffusers`](#diffusers) backend.

## diffusers

`diffusers` backend uses the huggingface [`diffusers`](https://huggingface.co/docs/diffusers/en/index) library to generate images. This backend only supports CUDA runtime.

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
