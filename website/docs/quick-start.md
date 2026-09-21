---
title: Quick start
---

Start with Qwen 3.5 4B for text chat on a CPU. Its Q4_K_M weights are about 2.74 GB; the runtime and context require additional memory.

:::note
The Qwen 3.5 image is pending staging validation and publication. The commands below use its planned tag. Until publication, use `ghcr.io/kaito-project/aikit/llama3.2:3b` with API model ID `llama-3.2-3b-instruct`. See the [model release process](./release.md#predefined-models).
:::

```bash
docker run -d --rm -p 8080:8080 ghcr.io/kaito-project/aikit/qwen3.5:4b
```

After running this, navigate to [http://localhost:8080/chat](http://localhost:8080/chat) to access the WebUI.

## API

AIKit provides an OpenAI-compatible endpoint.

For example:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "qwen-3.5-4b",
    "messages": [{"role": "user", "content": "explain kubernetes in a sentence"}]
  }'
```

Example response:

```jsonc
{
  // ...
    "model": "qwen-3.5-4b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Kubernetes is an open-source container orchestration system that automates the deployment, scaling, and management of applications and services, allowing developers to focus on writing code rather than managing infrastructure."
            }
        }
    ],
    // ...
}
```

This preset serves text and defaults to direct answers with reasoning disabled. See [pre-made models](./premade-models.md) for larger models and GPU options.

## Demo

https://www.youtube.com/watch?v=O0AOnxXp-o4

## What's next?

- Choose another [pre-made model](./premade-models.md), such as Gemma 4, Devstral Small 2, or GPT-OSS.
- [Create a custom model image](./create-images.md).
- [Fine-tune a model](./fine-tune.md) with domain-specific knowledge.
