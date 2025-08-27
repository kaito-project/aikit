---
title: Creating Model Images
---

:::note
This section shows how to create a custom image with models of your choosing. If you want to use one of the models from the model gallery, skip to [running models](#running-models).
:::

First, create a buildx buildkit instance.

```bash
docker buildx create --use --name aikit-builder
```

## Quick Start

You can easily build an image using the following ways:

### Hugging Face

🎬 Demo: https://www.youtube.com/watch?v=DI5NbdEFLC8

You can use [Hugging Face](https://huggingface.co) models directly by providing the model URL. For example:

```bash
docker buildx build -t my-model --load \
	--build-arg="model=huggingface://TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf" \
	"https://raw.githubusercontent.com/kaito-project/aikit/main/models/aikitfile.yaml"
```

Resulting model name will be the file name. In this case, `llama-2-7b-chat.Q4_K_M.gguf`.

:::tip
Syntax for Hugging Face source is `huggingface://{organization}/{repository}/{branch}/{file}`.

If the branch is `main`, it can be omitted (`huggingface://{organization}/{repository}/{file}`).
:::

### HTTP(S)

You can use HTTP(S) URLs to download models. For example:

```bash
docker buildx build -t my-model --load \
    --build-arg="model=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" \
    "https://raw.githubusercontent.com/kaito-project/aikit/main/models/aikitfile.yaml"
```

Resulting model name will be the file name. In this case, `llama-2-7b-chat.Q4_K_M.gguf`.

### OCI Artifacts

🎬 Demo: https://www.youtube.com/watch?v=G6PrzhEe_p8

You can use OCI artifacts to download models. For example:

```bash
docker buildx build -t my-model --load \
    --build-arg="model=oci://registry.ollama.ai/library/llama3:8b" \
    "https://raw.githubusercontent.com/kaito-project/aikit/main/models/aikitfile.yaml"
```

Resulting model name will be the image name. In this case, `llama3`.

After building the image, you can proceed to [running models](#running-models) to start the server.

### Build Arguments

Below are the build arguments you can use to customize the image:

#### `model`

The `model` build argument is the model URL to download and use. You can use any Hugging Face (`huggingface://`), HTTP(S) (`http://` or `https://`), or OCI (`oci://`). For example:

`--build-arg="model=huggingface://TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"`

#### `runtime`

The `runtime` build argument adds the applicable runtimes to the image. By default, aikit will automatically choose the most optimized CPU runtime.

You can use `cuda` to include NVIDIA CUDA runtime libraries. For example:

`--build-arg="runtime=cuda"`.

or `applesilicon` to include Apple Silicon runtime libraries. For example:

`--build-arg="runtime=applesilicon"`.

### Multi-Platform Support

AIKit supports AMD64 and ARM64 multi-platform images. To build a multi-platform image, you can simply add `--platform linux/amd64,linux/arm64` to the build command. For example:

```bash
docker buildx build -t my-model --load \
    --platform linux/amd64,linux/arm64 \
    --build-arg="model=huggingface://TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf" \
    "https://raw.githubusercontent.com/kaito-project/aikit/main/models/aikitfile.yaml"
```

[Model Gallery](https://kaito-project.github.io/aikit/docs/premade-models) is offered with multi-platform support. Docker runtime will automatically choose the correct platform to run the image. For more information, please see [multi-platform images documentation](https://docs.docker.com/build/building/multi-platform/).

:::note
Please note that ARM64 support only applies to the `llama.cpp` backend with CPU inference. NVIDIA CUDA is not supported on ARM64 at this time.
:::

## Advanced Usage

🎬 Demo: https://www.youtube.com/watch?v=5AQfG5VwN2c&list=PLx4Tje2rS923Bkw83GkobOyjIFLfxNrvs&index=2

Create an `aikitfile.yaml` with the following structure:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
models:
  - name: llama-2-7b-chat.Q4_K_M.gguf
    source: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

:::tip
For full `aikitfile` inference specifications, see [Inference API Specifications](docs/specs-inference.md).
:::

Then build your image with:

```bash
docker buildx build . -t my-model -f aikitfile.yaml --load
```

This will build a local container image with your model(s). You can see the image with:

```bash
docker images
REPOSITORY    TAG       IMAGE ID       CREATED             SIZE
my-model      latest    e7b7c5a4a2cb   About an hour ago   5.51GB
```

### Running models

You can start the inferencing server for your models with:

```bash
# for model gallery, replace "my-model" with the image name
docker run -d --rm -p 8080:8080 my-model
```

You can then send requests to `localhost:8080` to run inference from your models. For example:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "llama-2-7b-chat.Q4_K_M.gguf",
     "messages": [{"role": "user", "content": "explain kubernetes in a sentence"}]
   }'
```

Output should be similar to:

```json
{
    "created": 1701236489,
    "object": "chat.completion",
    "id": "dd1ff40b-31a7-4418-9e32-42151ab6875a",
    "model": "llama-2-7b-chat",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "\nKubernetes is a container orchestration system that automates the deployment, scaling, and management of containerized applications in a microservices architecture."
            }
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
```
