---
title: Inference API Specifications
---

## v1alpha1

```yaml
apiVersion: # required. only v1alpha1 is supported at the moment
debug: # optional. if set to true, debug logs will be printed
runtime: # optional. omit for the default CPU runtime. can be "cuda", "rocm", or "applesilicon"
backends: # optional. list of additional backends. can be "llama-cpp" (default), "diffusers", "vllm", "vllm-cpp"
loadToMemory: # optional. list of LocalAI model config names to load when the container starts
  - model-name
models: # optional. list of models to build. omit for runner mode (see runners.md)
  - name: # required. name of the model
    source: # required. source of the model. can be a url or a local file
    sha256: # optional. sha256 hash of the model file
    promptTemplates: # optional. list of prompt templates for a model
      - name: # required. name of the template
        template: # required. template string
config: # optional. list of config files
```

:::note
If omitted, `runtime` uses the default CPU runtime. `rocm` currently supports only the `llama-cpp` backend on `linux/amd64`. The experimental `vllm-cpp` backend supports CPU on `linux/amd64` and `linux/arm64`, or CUDA 13 on Blackwell-class `linux/amd64` GPUs.
:::

:::tip
When `backends` is specified without `models`, a **runner image** is created that downloads models at container startup. See [Runner Images](runners.md) for details.
:::

### Loading models at startup

`loadToMemory` opts models into loading before LocalAI starts serving requests:

```yaml
loadToMemory:
  - llama
```

Each item is an exact, case-sensitive LocalAI model-config name. For a baked `config`, use its `config[].name`; configs discovered or generated at runtime use the name declared in that config. Do not use a filename from the top-level `models` list. Names must be non-empty and unique, and cannot contain commas or backslashes. A model composed of several files, such as weight shards or a model plus an `mmproj` file, still has one logical name and needs one entry. Multiple names are loaded in the listed order.

Runner images generate their LocalAI config after resolving the runtime model argument. The `llama-cpp` runner derives the name from the selected GGUF filename without the `.gguf` extension; Diffusers and vLLM runners use the normalized final component of the model source. The `vllm-cpp` runner downloads a Hugging Face repository containing `config.json` and safetensors weights to a local model directory, or accepts a direct HTTP(S) URL ending in `.gguf`, because the native backend cannot download bare repository IDs itself. Use a direct file URL for GGUF weights. Avoid a baked `loadToMemory` setting when a runner source can resolve to several GGUF files because the selected name is ambiguous.

Loading blocks server startup and can fail if the model does not exist or there is insufficient memory, so health probes must allow enough startup time. It warms the model but does not prevent LocalAI from unloading it later, and LocalAI skips startup loading when `LOCALAI_SINGLE_ACTIVE_BACKEND=true`. The setting is disabled when omitted. At runtime, `LOCALAI_LOAD_TO_MEMORY` can replace the image default; set it to an empty value to disable startup loading.

Example:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
debug: true
runtime: cuda
loadToMemory:
  - llama-2-7b-chat
models:
  - name: llama-2-7b-chat
    source: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
    sha256: "08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa"
    promptTemplates:
      - name: "llama-2-7b-chat"
        template: |
          {{if eq .RoleName \"assistant\"}}{{.Content}}{{else}}
          [INST]
          {{if .SystemPrompt}}{{.SystemPrompt}}{{else if eq .RoleName \"system\"}}<<SYS>>{{.Content}}<</SYS>>

          {{else if .Content}}{{.Content}}{{end}}
          [/INST]
          {{end}}
config: |
  - name: \"llama-2-7b-chat\"
    backend: \"llama\"
    parameters:
      top_k: 80
      temperature: 0.2
      top_p: 0.7
      model: \"llama-2-7b-chat.Q4_K_M.gguf\"
    context_size: 4096
    roles:
      function: 'Function Result:'
      assistant_function_call: 'Function Call:'
      assistant: 'Assistant:'
      user: 'User:'
      system: 'System:'
    template:
      chat_message: \"llama-2-7b-chat\"
    system_prompt: \"You are a helpful assistant, below is a conversation, please respond with the next message and do not ask follow-up questions\"
```
