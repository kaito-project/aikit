---
title: Inference API Specifications
---

## v1alpha1

```yaml
apiVersion: # required. only v1alpha1 is supported at the moment
debug: # optional. if set to true, debug logs will be printed
runtime: # optional. defaults to avx. can be "avx", "avx2", "avx512", "cuda"
backends: # optional. list of additional backends. can be "llama-cpp" (default), "exllama2", "diffusers"
models: # required. list of models to build
  - name: # required. name of the model
    source: # required. source of the model. can be a url or a local file
    sha256: # optional. sha256 hash of the model file
    promptTemplates: # optional. list of prompt templates for a model
      - name: # required. name of the template
        template: # required. template string
config: # optional. list of config files
```

Example:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
debug: true
runtime: cuda
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
