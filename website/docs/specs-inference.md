---
title: Inference API Specifications
---

## v1alpha1

```yaml
apiVersion: # required. only v1alpha1 is supported at the moment
debug: # optional. if set to true, debug logs will be printed
runtime: # optional. omit for CPU; can be "cpu", "cuda", "cuda-12", "cuda-13", "rocm", or "applesilicon"
backends: # optional. list containing at most one family from the embedded catalog; omit for the catalog default
loadToMemory: # optional. list of LocalAI model config names to load when the container starts
  - model-name
models: # optional. list of models to embed. an explicit backend with no models requests runner mode (see runners.md)
  - name: # required. name of the model
    source: # required. source of the model. can be a url or a local file
    sha256: # optional. sha256 hash of the model file
    promptTemplates: # optional. list of prompt templates for a model
      - name: # required. name of the template
        template: # required. template string
config: # optional. inline LocalAI YAML configuration
```

:::note
If omitted, `runtime` uses CPU. Backend, runtime, and platform compatibility comes from the catalog embedded in the selected frontend release; the API does not maintain a separate hardcoded backend allowlist.
:::

:::tip
AIKit enters **runner mode** only when `backends` contains one family and `models` is empty. Every other inference build uses standard mode. Runner mode is available only when the resolved catalog entry has an explicit `runnerProfile`; this is catalog policy, not an aikitfile field. See [Runner Images](runners.md).
:::

### Backend catalog selection

`backends` selects one logical LocalAI family, while the optional `runtime` field selects its execution profile. Omitting `backends` uses the catalog's default family.

| `runtime` value | Requested profile |
|---|---|
| Omitted or `cpu` | CPU/default artifact. |
| `cuda` | CUDA 12; retained as the backward-compatible alias. |
| `cuda-12` | Exact CUDA 12 artifact. |
| `cuda-13` | Exact CUDA 13 artifact. |
| `rocm` | AMD ROCm artifact. |
| `applesilicon` | Apple Silicon ARM64 profile. |

`vulkan` and `intel` are not public runtime values. On Linux ARM64, a CUDA request resolves internally to the corresponding exact NVIDIA L4T artifact for the requested CUDA major. If that backend and platform do not have the corresponding artifact, the build fails.

Existing aikitfiles that omit `runtime` or use `runtime: cuda`, `runtime: rocm`, or `runtime: applesilicon` remain valid. `cpu`, `cuda-12`, and `cuda-13` add explicit spellings without changing those existing values.

Source compatibility does not guarantee identical image contents across frontend releases. A newer frontend can embed different artifact digests, defaults, statuses, or install instructions. Pin the frontend by digest when those choices must remain fixed.

The selected LocalAI version and immutable artifact references are recorded in the catalog plan and output metadata.

For every standard build, AIKit resolves one exact plan for each target platform. The plan, rather than backend-family branches in the frontend, supplies:

| Catalog field | Build behavior |
|---|---|
| `runtimeBase` | Default base image used by the resulting image. |
| `runnerRuntimeBase` | Optional runner-only base override when the runtime downloader needs package tooling. |
| `core` | LocalAI executable artifact. |
| `backend` and `fallbacks` | Primary backend and any explicitly declared companion backend artifacts. |
| `systemPackages` | OS package names installed from the runtime base's configured repositories for that exact tuple. |
| `runtimeSymlinks` | Compatibility symlinks created inside the runtime image for that exact tuple. |
| `environment` | Runtime environment added to the image. |
| `runnerProfile` | Explicit runner adapter, consulted only in runner mode. |

The runtime base, LocalAI core, and backend artifacts are all OCI digest-pinned. Package names in `systemPackages` are resolved at build time and are not version- or digest-locked by the catalog. Any `supported` or `experimental` catalog tuple can be materialized in standard mode, including families not covered by a dedicated AIKit guide. Generic installation does not prove that a particular model format or LocalAI configuration works end to end; the model and `config` must still match the selected LocalAI backend.

Use an explicit versioned CUDA runtime when the CUDA major must be fixed:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
runtime: cuda-13
backends:
  - llama-cpp
# This build succeeds only if the selected frontend contains this exact tuple.
models:
  - name: model.gguf
    source: https://example.com/model.gguf
```

Runtime availability is release-specific. The generated catalog lock embedded in the selected frontend is authoritative; a documentation example does not guarantee that the tuple exists in every release.

Catalog entries have one of these statuses:

| Status | Build behavior |
|---|---|
| `supported` | Selectable for the exact family, runtime, and platform tuple under AIKit's current support policy. |
| `experimental` | Selectable for the exact tuple, but has less compatibility assurance and may change. |
| `quarantined` | Disabled by catalog policy; builds cannot select it. |
| `deprecated` | Retained as catalog history but unavailable for new builds. |

Status describes tuple selection, not end-to-end validation of every workload, and it does not make a tuple runner-capable. Runner mode separately requires `runnerProfile`.

AIKit does not silently substitute another family, CUDA major, runtime, or platform. A missing, incompatible, quarantined, or deprecated selection fails the build. In particular, `cuda-12` never falls forward to CUDA 13, `cuda-13` never falls back to CUDA 12, and an unavailable CUDA ARM64/L4T artifact is not replaced with an AMD64 artifact. For a multi-platform build, every requested platform must resolve before the build proceeds.

Catalog `fallbacks` are different from selection fallback. They are digest-pinned companion artifacts deliberately installed by the selected plan—for example, a CUDA llama.cpp plan can include its CPU backend for runtime use without a GPU. AIKit never reaches for an undeclared artifact or changes the requested tuple to make resolution succeed.

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
