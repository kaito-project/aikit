---
title: llama.cpp (GGUF and GGML)
---

AIKit's default catalog family is `llama-cpp`, backed by [llama.cpp](https://github.com/ggerganov/llama.cpp) through LocalAI. It provides inference for LLaMA and many other model architectures in C/C++.

No `backends` field is required to select the catalog default.

This backend:

- provides support for GGUF (recommended) and GGML models
- has catalog plans for CPU and accelerator runtimes on selected platforms

Exact runtime, platform, and status availability is defined by the catalog embedded in the selected frontend release. CPU dispatch such as AVX2 happens inside the selected LocalAI backend. Use `runtime: cuda-12` or `runtime: cuda-13` to request a CUDA major; on Linux ARM64, AIKit selects the corresponding L4T artifact internally. See [Backend catalog selection](specs-inference.md#backend-catalog-selection).

## Example

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the examples below.
:::

### CPU
https://github.com/kaito-project/aikit/blob/main/test/aikitfile-llama.yaml

### GPU (NVIDIA CUDA)
https://github.com/kaito-project/aikit/blob/main/test/aikitfile-llama-cuda.yaml

### GPU (ROCm)
https://github.com/kaito-project/aikit/blob/main/test/aikitfile-llama-rocm.yaml
