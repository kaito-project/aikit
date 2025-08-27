---
title: Introduction
slug: /
---

AIKit is a comprehensive platform to quickly get started to host, deploy, build and fine-tune large language models (LLMs).

AIKit offers two main capabilities:

- **Inference**: AIKit uses [LocalAI](https://localai.io/), which supports a wide range of inference capabilities and formats. LocalAI provides a drop-in replacement REST API that is OpenAI API compatible, so you can use any OpenAI API compatible client, such as [Kubectl AI](https://github.com/sozercan/kubectl-ai), [Chatbot-UI](https://github.com/sozercan/chatbot-ui) and many more, to send requests to open LLMs!

- **[Fine Tuning](fine-tune.md)**: AIKit offers an extensible fine tuning interface. It supports [Unsloth](https://github.com/unslothai/unsloth) for fast, memory efficient, and easy fine-tuning experience.

👉 To get started, please see [Quick Start](quick-start.md)!

## Features

- 💡 No GPU, or Internet access is required for inference!
- 🐳 No additional tools are needed except for [Docker](https://docs.docker.com/desktop/install/linux-install/) or [Podman](https://podman.io)!
- 🤏 Minimal image size, resulting in less vulnerabilities and smaller attack surface with a custom [chiseled](https://ubuntu.com/containers/chiseled) image
- 🎵 [Fine tune support](fine-tune.md)
- 🚀 Easy to use declarative configuration for [inference](specs-inference.md) and [fine tuning](specs-finetune.md)
- ✨ OpenAI API compatible to use with any OpenAI API compatible client
- 📸 [Multi-modal model support](vision.md)
- 🖼️ [Image generation support](diffusion.md)
- 🦙 Support for GGUF ([`llama`](https://github.com/ggerganov/llama.cpp)), GPTQ or EXL2 ([`exllama2`](https://github.com/turboderp/exllamav2)), and GGML ([`llama-ggml`](https://github.com/ggerganov/llama.cpp)) models
- 🚢 [Kubernetes deployment ready](#kubernetes-deployment)
- 📦 Supports multiple models with a single image
- 🖥️ Supports [AMD64 and ARM64](create-images.md#multi-platform-support) CPUs and [GPU-accelerated inferencing with NVIDIA GPUs](gpu.md)
- 🔐 Ensure [supply chain security](security.md) with SBOMs, Provenance attestations, and signed images
- 🌈 Support for non-proprietary and self-hosted container registries to store model images
