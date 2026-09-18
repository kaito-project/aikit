---
title: llama.cpp (GGUF and GGML)
---

AIKit's default catalog family is `llama-cpp`, backed by [llama.cpp](https://github.com/ggerganov/llama.cpp) through LocalAI. It provides inference for LLaMA and many other model architectures in C/C++.

No `backends` field is required to select the catalog default.

This backend:

- provides support for GGUF (recommended) and GGML models
- has catalog plans for CPU and accelerator runtimes on selected platforms

Exact runtime, platform, and status availability is defined by the catalog embedded in the selected frontend release. CPU dispatch such as AVX2 happens inside the selected LocalAI backend. Use `runtime: cuda-12` or `runtime: cuda-13` to request a CUDA major; on Linux ARM64, AIKit selects the corresponding L4T artifact internally. See [Backend catalog selection](specs-inference.md#backend-catalog-selection).

## Qwen3-TTS

The [Qwen3-TTS aikitfile](https://github.com/kaito-project/aikit/blob/main/models/qwen3-tts-1.7b-base.yaml) packages the 1.7B Base model at Q4_K_M and its Q8_0 audio projector. It uses the development frontend with LocalAI v4.10.0 and defaults to CPU. Both downloads are pinned by revision and SHA-256.

Build the model image from the repository root:

```bash
docker buildx build --load -t qwen3-tts \
  -f models/qwen3-tts-1.7b-base.yaml .
```

The Base model requires reference audio. Place a short WAV voice recording at `reference.wav` in the repository root, then mount it when starting the server:

```bash
docker run -d --rm --name qwen3-tts \
  -p 127.0.0.1:8080:8080 \
  --mount "type=bind,source=$(pwd)/reference.wav,target=/models/reference.wav,readonly" \
  qwen3-tts
```

After the server starts, generate speech through LocalAI's `/v1/audio/speech` endpoint. The `voice` value is the reference file's path inside the container:

```bash
curl --fail http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello from AIKit.",
    "voice": "/models/reference.wav",
    "response_format": "wav"
  }' \
  --output speech.wav
```

This uses LocalAI's [Qwen3-TTS integration for llama.cpp](https://github.com/mudler/LocalAI/blob/v4.10.0/gallery/index.yaml#L18248). The model and projector are baked into the image; the reference recording is supplied at runtime.

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
