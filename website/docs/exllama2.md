---
title: Exllama v2 (GPTQ and EXL2)
---

[ExLlamaV2](https://github.com/turboderp/exllamav2) is an inference library for running local LLMs on modern consumer GPUs.

This backend:
- provides support for GPTQ and EXL2 models
- requires CUDA runtime

:::note
This is an experimental backend and it may change in the future.
:::

## Example

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the examples below.
:::

### EXL2
https://github.com/kaito-project/aikit/blob/main/test/aikitfile-exllama2-exl2.yaml

### GPTQ
https://github.com/kaito-project/aikit/blob/main/test/aikitfile-exllama2-gptq.yaml
