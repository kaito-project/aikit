---
title: Fine Tuning
---

Fine tuning process allows the adaptation of pre-trained models to domain-specific data. At this time, AIKit fine tuning process is only supported with NVIDIA GPUs; AMD ROCm is not supported yet.

:::note
Fine tuning requires Docker Engine 27 or later, Buildx 0.22 or later, BuildKit 0.20 or later, and NVIDIA Container Toolkit configured with CDI support. AIKit requests the first GPU as `nvidia.com/gpu=0` by default for its training and export build steps instead of creating NVIDIA device nodes or installing a matching driver inside the build container. Supply an immutable `GPU-...` or `MIG-...` selector through `cdiDevice` when cross-session GPU result-cache reuse is required.

On Windows Subsystem for Linux, use Docker Desktop with WSL2 GPU paravirtualization and Buildx 0.27 or later. Earlier Buildx versions do not expose the WSL driver libraries to a containerized GPU builder. See Docker's [CDI build documentation](https://docs.docker.com/build/building/cdi/).

Verify the host prerequisites before creating the builder:

```bash
nvidia-smi
nvidia-ctk cdi list
docker version
docker buildx version
```
:::

## Getting Started

To get started, you need to create a builder to be able to access host GPU devices.

Create a builder with the following configuration:

```bash
docker buildx create --name aikit-builder --use \
  --driver docker-container \
  --driver-opt image=moby/buildkit:buildx-stable-1-gpu \
  --buildkitd-flags '--allow-insecure-entitlement device'
docker buildx inspect aikit-builder --bootstrap
```

The GPU-capable BuildKit image is experimental. The inspection output must list a usable NVIDIA selector such as `nvidia.com/gpu=0`, an immutable UUID selector, or the on-demand NVIDIA device kind `nvidia.com/gpu` under `Devices`. If it does not, ensure the NVIDIA CDI specification and hook are available to the BuildKit daemon before continuing. The standard CDI directories are `/etc/cdi`, `/var/run/cdi`, and `/etc/buildkit/cdi`.

:::tip
Additionally, you can build using other BuildKit drivers, such as [Kubernetes driver](https://docs.docker.com/build/drivers/kubernetes/) by setting `--driver=kubernetes` if you are interested in building using a Kubernetes cluster. Please see [BuildKit Drivers](https://docs.docker.com/build/drivers/) for more information.
:::

## Targets and Configuration

AIKit is capable of supporting multiple fine tuning implementation targets. At this time, [Unsloth](https://github.com/unslothai/unsloth) is the only supported target, but can be extended for other fine tuning implementations in the future.

### Unsloth

Create a YAML file with your configuration. For example, minimum config looks like:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
baseModel: "unsloth/llama-2-7b-bnb-4bit" # base model to be fine tuned. this can be any model from Huggingface. For unsloth optimized base models, see https://huggingface.co/unsloth
datasets:
  - source: "yahma/alpaca-cleaned" # Hugging Face dataset identifier or an HTTP(S) URL
    type: "alpaca" # record schema: alpaca, messages, sharegpt, prompt-completion, text, or preference
config:
  unsloth:
```

For full configuration, please refer to [Fine Tune API Specifications](./specs-finetune.md).

#### Choose SFT or DPO

AIKit's Unsloth target supports supervised fine-tuning (SFT) and Direct Preference Optimization (DPO). Both objectives train from fixed datasets. DPO is offline preference optimization, not an online reinforcement-learning environment loop: AIKit does not collect live rewards or run policy rollouts against an environment.

| Objective | What the model learns from | Compatible record schemas | Dataset count |
| --- | --- | --- | --- |
| SFT (default) | Demonstrated outputs or complete training sequences | `alpaca`, `messages`, `sharegpt`, `prompt-completion`, or `text` | One or more mutually compatible datasets |
| DPO | A `chosen` response compared with a `rejected` response for the same prompt | `preference` | Exactly one dataset |

The training objective, record schema, and loader are independent choices:

| Setting | Selects | Examples |
| --- | --- | --- |
| `objective.type` | Training objective | `sft`, `dpo` |
| `datasets[].type` | Record schema, required fields, and loss semantics | `messages`, `prompt-completion`, `preference` |
| `datasets[].loader.type` | Source transport and file parser | `huggingface`, `json`, `csv`, `parquet`, `text` |

For example, `type: preference` describes records with `prompt`, `chosen`, and `rejected` fields, while `loader.type: json` says only that those records are read from JSON. A Parquet file can contain the same `preference` schema. The `text` record schema and the `text` loader are also distinct: the schema requires a `text` field, while the loader turns each line of a remote text file into a `text` record.

If `objective` is omitted, YAML `null`, an empty mapping, or explicitly set to `sft`, AIKit uses SFT and defaults `learningRate` to `0.0002`. Set `objective.type: dpo` to optimize explicit preferences:

```yaml
objective:
  type: dpo
  beta: 0.1
  lossType: sigmoid
  maxPromptLength: 512
datasets:
  - source: organization/preferences
    type: preference
    loader:
      type: huggingface
      split: train
      revision: 0123456789abcdef0123456789abcdef01234567
config:
  unsloth:
    packing: false
    maxSeqLength: 2048
```

DPO defaults to `beta: 0.1`, `lossType: sigmoid`, `maxPromptLength: 512`, and `learningRate: 0.000001`. `beta` must be finite and positive, `maxPromptLength` must not exceed `maxSeqLength`, and only sigmoid loss is supported. DPO requires exactly one `preference` dataset, requires `packing: false`, rejects every SFT record schema, and rejects the SFT-only `loss: response` setting. Before training, AIKit rejects a pair when its chosen and rejected responses become token-identical after the trainer's effective tokenization and pinned `keep_end` truncation, because that pair would provide no preference signal.

AIKit constructs a separate DPO trainer with the LoRA policy and `ref_model=None`. The same PEFT model with its adapter disabled supplies reference log probabilities; this is not reference-free DPO. Preference records bypass all SFT formatting and masking paths. DPO and SFT use the same trained-adapter save path and can both produce either a GGUF or adapter output.

#### Dataset Formats

| Record schema (`datasets[].type`) | Expected records | Objective and loss behavior |
| --- | --- | --- |
| `alpaca` | `instruction`, `input`, and `output` strings | SFT full-sequence |
| `messages` | Canonical `role`/`content` conversations | SFT full-sequence with `loss: all`; assistant-response-only with `loss: response` |
| `sharegpt` | ShareGPT `from`/`value` conversations | SFT full-sequence with `loss: all`; assistant-response-only with `loss: response` |
| `prompt-completion` | Separate `prompt` and non-empty `completion` strings | SFT completion-only; prompt tokens are masked |
| `text` | A complete preformatted sequence in `text` | SFT full-sequence |
| `preference` | Explicit non-empty `prompt`, `chosen`, and `rejected` strings | DPO only; chosen and rejected responses must remain distinct after effective truncation |

| Loader (`datasets[].loader.type`) | Source and parsing behavior |
| --- | --- |
| omitted | Legacy behavior: HTTP(S) uses JSON; other sources are passed to Hugging Face Datasets |
| `huggingface` | A Hugging Face dataset identifier, with optional subset, split, and pinned revision |
| `json`, `csv`, or `parquet` | An HTTP(S) file parsed with the corresponding builder |
| `text` | An HTTP(S) text file; each line becomes one record with a `text` field |

#### Dataset Loading and Reproducibility

When `loader` is omitted, AIKit preserves the original behavior: HTTP(S) sources use the JSON builder with the `train` split, and every other source is passed to Hugging Face Datasets with the `train` split. These mutable sources remain supported, but AIKit emits a reproducibility warning because their bytes are not pinned.

Use the `huggingface` loader to select a Hub subset, split, and immutable revision:

```yaml
datasets:
  - source: HuggingFaceH4/ultrachat_200k
    type: messages
    loader:
      type: huggingface
      subset: default
      split: train_sft
      revision: 0123456789abcdef0123456789abcdef01234567
```

A supplied `revision` must be a lowercase 40-character commit hash. Branches, tags, and short hashes are rejected because they are mutable. Omitting `revision` is allowed and emits a warning.

Use `json`, `csv`, `parquet`, or `text` for an HTTP(S) file. A checksum pins the raw downloaded bytes before parsing or decompression:

```yaml
datasets:
  - source: https://datasets.example.com/train.parquet
    type: prompt-completion
    loader:
      type: parquet
      split: train
      checksum: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Remote files are downloaded into an AIKit-owned content-addressed cache under the persistent Hugging Face Datasets cache. AIKit verifies cached and newly downloaded bytes before invoking the selected Datasets builder or allocating the model, and remote file handling limits the caller's wait for each open or blocking read to 60 seconds, caps the caller's aggregate network wait at one hour, and caps raw transferred and cached bytes at 64 GiB before parsing or decompression. These fixed limits are not configurable. Cache publication and private verified snapshots require additional temporary disk space, and decompressed dataset size is not currently bounded. A missing checksum remains allowed but emits a warning and cannot make a BuildKit cache entry immutable. Dataset sources must not contain leading or trailing whitespace. URL credentials, query values, and fragments are not included in AIKit-generated errors or warning logs, cache filenames, or cache metadata. The configured source URL is still part of the training configuration and BuildKit definition, so credential-bearing URLs are not a supported secret mechanism; private-dataset secret mounts remain out of scope.

The loader `split` defaults to `train` and must contain letters, numbers, or underscores in one or more dot-separated segments. It selects the training split only; it does not configure evaluation data or metrics. The `text` loader turns each input line into a `text` record and therefore cannot provide DPO preference columns. JSON, CSV, and Parquet loaders can provide any record schema compatible with the selected objective when the required columns and values are present. Unknown fields inside `loader` fail instead of being silently ignored.

#### Combining Multiple SFT Datasets

AIKit supports one or more datasets for SFT. DPO is intentionally separate and requires exactly one `preference` dataset. All entries in an SFT job must resolve to the same SFT mode:

| SFT mode | Compatible dataset combination |
| --- | --- |
| Full-sequence | Any mix of `alpaca`, `text`, `messages`, and `sharegpt` with global `loss: all` and compatible tokenizer special-token boundaries |
| Completion-only | One or more `prompt-completion` datasets |
| Response-only chat | Any mix of `messages` and `sharegpt` with global `loss: response` and `packing: false` |

Mixing modes is unsupported. For example, `alpaca` cannot be combined with `prompt-completion`, and response-only chat cannot be combined with `text`. `config.unsloth.loss` applies to the whole job; per-dataset loss, weighted sampling, and random interleaving are not supported.

AIKit loads sources sequentially in YAML order, validates every source as nonempty, normalizes each source independently, and concatenates records in configured order while preserving row order within each source. This deterministic order is the input to the SFT trainer; packing, shuffling, or sampling may reorder records afterward. Datasets are not deduplicated, over- or undersampled, so repeating an entry intentionally repeats its rows. A one-source configuration retains the existing single-source path.

The locked Unsloth trainer selects one rendered-text `add_special_tokens` setting from the first canonical record. AIKit verifies that every record in a combined full-sequence job produces the same token IDs under that dataset-wide setting as it would under its source-specific setting. If, for example, rendered chat records already contain BOS while Alpaca records rely on the tokenizer to add BOS, AIKit rejects the mix with the failing `datasets[n]` index instead of allowing dataset order to duplicate or omit BOS tokens. A setting difference that does not change token IDs remains valid; reordering sources cannot bypass a real boundary mismatch.

Multiple `prompt-completion` sources receive the same protection. AIKit compares the effective token IDs and completion-mask boundary under each source's own special-token policy with the dataset-wide policy selected from the first source. It rejects only differences that change trained tokens or which tokens are treated as the prompt; no-op policy differences remain valid.

This example combines pinned Alpaca and preformatted-text sources in the full-sequence group while using different loaders:

```yaml
datasets:
  - source: organization/instruction-data
    type: alpaca
    loader:
      type: huggingface
      split: train
      revision: 0123456789abcdef0123456789abcdef01234567
  - source: https://datasets.example.com/domain-text.jsonl
    type: text
    loader:
      type: json
      split: train
      checksum: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
config:
  unsloth:
    loss: all
```

Each entry retains its own loader, split, subset, revision, and checksum identity. Changing or reordering any entry invalidates training and export while leaving dependency installation cacheable. Incompatible entries fail with the relevant `datasets[n]` indexes instead of being coerced or ignored.

#### Record Schema Details

For the chat dataset types `messages` and `sharegpt`, `config.unsloth.loss` controls which tokens are supervised:

- `all` is the default when `loss` is omitted or set to YAML `null` and preserves full-sequence chat training. System, user, assistant, and chat-template tokens are supervised.
- `response` supervises assistant responses and masks the rest of each rendered conversation.

```yaml
config:
  unsloth:
    loss: response
```

An explicit empty `loss` string is invalid. Response-only loss is not supported for `alpaca`, `prompt-completion`, or `text`; those types retain their established loss behavior. For the initial response-only release, `packing` must be `false` so masking cannot cross conversation boundaries. AIKit derives response markers from the model's deterministic chat template, rejects marker collisions with message content or rendered role boundaries, applies Unsloth's response masking after constructing the trainer, and validates the actual prepared labels before training. Missing or unusable markers, a prepared dataset with no supervised response tokens, or labels that do not match the expected response spans fail instead of falling back to full-sequence loss. Custom templates and marker strings are not accepted.

##### Alpaca

The `alpaca` type uses the existing `instruction`, `input`, and `output` string columns. AIKit renders the existing Alpaca prompt, appends EOS explicitly, and trains with full-sequence loss. Existing Alpaca configurations require no changes.

##### Prompt-Completion

The `prompt-completion` type keeps the `prompt` and `completion` columns separate. Every record must contain a string `prompt` and a non-empty string `completion`. Prompt tokens are masked from the loss, while completion and EOS tokens are supervised. Empty datasets and missing, null, or invalid fields are rejected before model allocation.

```yaml
datasets:
  - source: organization/question-answer-data
    type: prompt-completion
```

An expected JSON Lines record is:

```json
{"prompt":"Question: What is a container image?\nAnswer:","completion":" An immutable package containing an application and its dependencies."}
```

##### Messages

The `messages` type accepts canonical text-only chat conversations and applies the base model tokenizer's existing chat template. Each record must contain a non-empty `messages` list. Benign top-level metadata columns, such as IDs and source labels, are ignored when the dataset is projected to `messages`; top-level tool, template-control, document, image, audio, and video fields are rejected. Every turn must be a mapping with exactly the string fields `role` and `content`; supported roles are `system`, `user`, and `assistant`. A conversation must contain at least one assistant turn and end with an assistant turn. Extra fields within turns, tool calls, unsupported roles, and structured or multimodal content are rejected instead of being passed to the tokenizer.

```yaml
datasets:
  - source: organization/chat-data
    type: messages
```

An expected JSON Lines record is:

```json
{"messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"What is a container image?"},{"role":"assistant","content":"An immutable application package."}]}
```

AIKit requires the tokenizer to provide a usable, deterministic chat template. Wall-clock-dependent templates containing `strftime_now` are rejected until deterministic template values can be configured and included in cache keys. Before allocating LoRA adapters, AIKit renders each conversation to the canonical `text` field with `tokenize=False` and `add_generation_prompt=False`. It does not add special tokens around the rendered text. AIKit verifies that the locked Unsloth text path produces the same token IDs as direct chat-template tokenization and rejects mismatches or sequences longer than `maxSeqLength` instead of truncating them. Validation errors include source and row context while redacting URL credentials and query values.

With the default `loss: all`, the `messages` type uses full-sequence SFT and retains existing packing and role-order behavior. Set `loss: response` to supervise assistant responses only; response-only configurations must set `packing: false`, place any system messages before the conversation, and then alternate user and assistant messages in complete pairs. AIKit verifies that derived marker tokens uniquely match the rendered user and assistant boundaries and do not collide with message content, then verifies the prepared labels before training. Tools, custom chat templates, and multimodal messages are not supported.

##### ShareGPT

The `sharegpt` type is a deterministic compatibility adapter for text-only ShareGPT records. Each record must contain a non-empty `conversations` list, and each turn must provide string `from` and `value` fields. AIKit uses this fixed role map:

- `system` becomes `system`.
- `human` and `user` become `user`.
- `gpt` and `assistant` become `assistant`.

Unknown roles, missing fields, and non-string values are rejected. AIKit does not infer alternate keys or roles. After conversion, the conversation follows the same validation, chat-template rendering, token-equivalence, and `loss` pipeline as `messages`; it must contain an assistant response and end with an assistant turn. Packing remains available with `loss: all` but is rejected with `loss: response`. A valid dataset may contain a single record, and benign top-level metadata does not affect the conversion.

```yaml
datasets:
  - source: organization/sharegpt-data
    type: sharegpt
config:
  unsloth:
    loss: response
```

An expected JSON Lines record is:

```json
{"conversations":[{"from":"system","value":"You are concise."},{"from":"human","value":"What is a container image?"},{"from":"gpt","value":"An immutable application package."}]}
```

##### Text

The `text` type accepts complete, preformatted training sequences in a non-empty string `text` column. AIKit preserves the sequence content while normalizing its special-token boundaries: tokenization produces exactly one effective leading BOS where applicable and every record ends with exactly one EOS. Tokenizers that add BOS or EOS automatically, as well as tokenizers that do not define BOS, are supported, but the tokenizer must define a usable EOS token. Records whose normalized token sequence exceeds `maxSeqLength` are rejected rather than truncated so that their terminal EOS is retained. All tokens in the normalized record are supervised, and packing retains the tokenized EOS record boundaries.

```yaml
datasets:
  - source: organization/domain-corpus
    type: text
```

An expected JSON Lines record is:

```json
{"text":"Question: What is a container image?\nAnswer: An immutable package containing an application and its dependencies."}
```

:::caution
The `text` type performs full-sequence supervised fine-tuning (SFT). It is not continued pretraining: AIKit retains the standard LoRA targets and optimizer configuration and does not train embedding or language-model-head parameters.
:::

##### Preference (DPO)

The `preference` type is valid only with `objective.type: dpo`. Every record must contain explicit, non-empty string `prompt`, `chosen`, and `rejected` values, and the chosen and rejected responses must differ. AIKit does not infer a prompt from response prefixes, accept conversational preference arrays, or normalize preference text.

```yaml
objective:
  type: dpo
  maxPromptLength: 96
datasets:
  - source: https://datasets.example.com/preferences.jsonl
    type: preference
    loader:
      type: json
      split: train
      checksum: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
config:
  unsloth:
    packing: false
    maxSeqLength: 256
```

An expected JSON Lines record is:

```json
{"prompt":"How should I rotate an API key?","chosen":"Deploy a replacement before revoking the old key.","rejected":"Revoke the old key before creating a replacement."}
```

AIKit drops unrelated metadata columns and passes the three preference strings to the DPO trainer without SFT formatting or EOS-text appending. It validates the trainer-prepared prompt and completion token sequences and rejects chosen/rejected completions that collapse to the same effective tokens after pinned `keep_end` truncation. Multiple preference datasets, implicit prompts, conversational preference arrays, evaluation datasets, alternate DPO losses, custom reference models, reference-free DPO, and online environment interaction are not supported.

:::note
Please refer to [Unsloth documentation](https://github.com/unslothai/unsloth) for more information about Unsloth configuration.
:::

#### Example Configurations

:::warning
Please make sure to change syntax to `#syntax=ghcr.io/kaito-project/aikit/aikit:latest` in the example below.
:::

- [Alpaca](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth.yaml)
- [Adapter output smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-adapter-smoke.yaml)
- [Messages smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-messages-smoke.yaml)
- [Response-only messages smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-messages-response-smoke.yaml)
- [Response-only ShareGPT smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-sharegpt-response-smoke.yaml)
- [Prompt-completion smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-prompt-completion-smoke.yaml)
- [Text smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-text-smoke.yaml)
- [Checksummed Parquet loader smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-loader-smoke.yaml)
- [Multiple compatible datasets smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-multiple-datasets-smoke.yaml)
- [Checksummed DPO preference smoke test](https://github.com/kaito-project/aikit/blob/main/test/aikitfile-unsloth-dpo-smoke.yaml)


## Build

Build using following command and make sure to replace `--target` with the fine-tuning implementation of your choice (`unsloth` is the only option supported at this time), `--file` with the path to your configuration YAML and `--output` with the output directory of the fine-tuned artifact. This example assumes the builder and Docker runtime use the same local NVIDIA device namespace. For a remote or on-demand builder, select a device shown by `docker buildx inspect` and omit `nvidiaDriverVersion` unless its driver version is known.

```bash
NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
NVIDIA_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)
NVIDIA_CDI_DEVICE="nvidia.com/gpu=${NVIDIA_GPU_UUID}"

docker buildx build --builder aikit-builder \
  --allow "device=${NVIDIA_CDI_DEVICE}" \
  --build-arg "nvidiaDriverVersion=${NVIDIA_DRIVER_VERSION}" \
  --build-arg "cdiDevice=${NVIDIA_CDI_DEVICE}" \
  --file "/path/to/config.yaml" \
  --output "/path/to/output" \
  --target unsloth \
  --progress plain .
```

The `device` entitlement authorizes only the CDI device requested by AIKit. Training and GGUF export run with the normal BuildKit sandbox security mode. The `cdiDevice` argument selects that GPU, while the optional `nvidiaDriverVersion` argument is used only as a GPU-phase cache discriminator; AIKit does not install that driver in the build container. AIKit reuses GPU results across BuildKit sessions only when the driver version is paired with an immutable `GPU-...` or `MIG-...` CDI selector. Index, `all`, and on-demand selectors can identify different hardware on another builder, so AIKit falls back to the current BuildKit session as the result-cache discriminator while retaining persistent model, dataset, and compiler caches.

Training and GGUF export are cached separately. Changing only `output.name` reuses training and, for GGUF, export. Changing only `output.quantize` reuses training and reruns GGUF export. Adapter output skips GGUF export entirely; switching from adapter to GGUF can reuse the cached training result.

### Choose an output format

`output.format` defaults to `gguf`, preserving the existing behavior. GGUF output is a merged, standalone model file named from `output.name` and `output.quantize`:

```yaml
output:
  format: gguf
  quantize: q4_k_m
  name: aikit-model
```

```bash
$ ls -al _output
-rw-r--r--  1 kaito-project kaito-project 7161089856 Mar  3 00:19 aikit-model-q4_k_m.gguf
```

Use `format: adapter` to export only the trained LoRA adapter and tokenizer assets:

```yaml
output:
  format: adapter
  name: aikit-adapter
```

For adapter output, omit `output.quantize` entirely; specifying it, including as YAML `null`, is rejected because no GGUF quantization occurs. The local output is a directory named exactly `output.name`:

```text
_output/
└── aikit-adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── optional tokenizer assets and templates
```

The adapter bundle intentionally excludes both GGUF and base-model weights. Its `adapter_config.json` records the exact resolved base-model repository and immutable revision loaded for training. When `loadIn4bit` is enabled, AIKit quantizes that pinned base while loading it rather than substituting a separately versioned prequantized repository. Load the recorded snapshot separately in a PEFT-capable consumer, then apply the adapter and bundled tokenizer configuration. Architecture, target modules, tokenizer behavior, and the runtime's LoRA support must all be compatible; a similarly named or newer base revision is not assumed compatible.

AIKit's current model-image flow consumes standalone GGUF files and does not directly serve this adapter directory. Choose GGUF for that path. Choose adapter when another PEFT-compatible runtime or downstream merge process will combine the base and LoRA weights; the base model remains subject to its own availability and license terms.

## Demo

https://www.youtube.com/watch?v=FZuVb-9i-94

## What's next?

👉 For GGUF output, refer to [Creating Model Images](./create-images.md) to create an AIKit image. Adapter output must instead be loaded with its compatible base model by a PEFT-capable consumer.

## Troubleshooting

### Build fails with `failed to solve: DeadlineExceeded: context deadline exceeded`

This is a known issue with BuildKit and might be related to disk speed. For more information, please see https://github.com/moby/buildkit/issues/4327

### Build fails because the requested NVIDIA CDI device is not registered

Run `nvidia-ctk cdi list` on the host and `docker buildx inspect aikit-builder --bootstrap`. The requested selector must be available in the environment where it is used. A local builder can use the same UUID listed by the host, while an on-demand builder may initially expose only `nvidia.com/gpu`. If no usable selector appears in the builder, make the NVIDIA CDI specification and hook available to the BuildKit daemon and recreate or restart the builder.

### Build fails with `requested by the build but not allowed`

Enable the `device` entitlement on the BuildKit daemon with `--allow-insecure-entitlement device`, and pass `--allow "device=${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=0}"` to the build command.
