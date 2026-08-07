---
title: Fine Tuning API Specifications
---

## v1alpha1

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: # required. only v1alpha1 is supported at the moment
baseModel: # required. any base model from Huggingface. for unsloth, see for 4bit pre-quantized models: https://huggingface.co/unsloth
objective: # optional. omission or null defaults to SFT
  type: # optional. defaults to sft; dpo selects Direct Preference Optimization
  beta: # DPO only. optional finite positive value; defaults to 0.1
  lossType: # DPO only. optional; only sigmoid is supported and is the default
  maxPromptLength: # DPO only. optional; defaults to 512 and must not exceed maxSeqLength
datasets:
  - source: # required. a Hugging Face dataset identifier or an absolute HTTP(S) URL
    type: # required record schema. can be "alpaca", "messages", "sharegpt", "prompt-completion", "text", or "preference"
    loader: # optional. omission preserves automatic legacy loading and the train split
      type: # required when loader is present. huggingface, json, csv, parquet, or text
      subset: # optional. Hugging Face loader only
      split: # optional. defaults to train; selects training data, not evaluation data
      revision: # optional. Hugging Face loader only; lowercase 40-character commit hash
      checksum: # optional. remote-file loaders only; sha256:<64 lowercase hex> of raw downloaded bytes
config:
  unsloth:
    loss: # optional SFT setting. omitted or null defaults to all. response is supported only for messages and sharegpt and rejected for DPO
    packing: # optional. defaults to false. not supported with loss: response or objective.type: dpo
    maxSeqLength: # optional. defaults to 2048
    loadIn4bit: # optional. defaults to true
    batchSize: # optional. default to 2
    gradientAccumulationSteps: # optional. defaults to 4
    warmupSteps: # optional. defaults to 10
    maxSteps: # optional. defaults to 60
    learningRate: # optional. defaults to 0.0002 for SFT and 0.000001 for DPO
    loggingSteps: # optional. defaults to 1
    optimizer: # optional. defaults to adamw_8bit
    weightDecay: # optional. defaults to 0.01
    lrSchedulerType: # optional. defaults to linear
    seed: # optional. defaults to 42
output:
  quantize: # optional. defaults to q4_k_m. for unsloth, see for allowed quantization methods: https://github.com/unslothai/unsloth/wiki#saving-to-gguf.
  name: # optional. defaults to "aikit-model"
```

### Training Objectives and Compatibility

The Unsloth target supports two offline training objectives: SFT and DPO. SFT learns from demonstrated outputs or complete sequences. DPO learns a relative preference between `chosen` and `rejected` responses for the same prompt. DPO is preference optimization, not an online reinforcement-learning environment loop; this API does not define environment interaction, policy rollouts, or live reward collection.

The configuration has three independent layers:

| Layer | Field | Allowed values | Contract |
| --- | --- | --- | --- |
| Training objective | `objective.type` | `sft`, `dpo` | Selects the SFT or DPO trainer. |
| Record schema | `datasets[].type` | `alpaca`, `messages`, `sharegpt`, `prompt-completion`, `text`, `preference` | Defines required record fields and loss semantics. |
| Loader/parser | `datasets[].loader.type` | `huggingface`, `json`, `csv`, `parquet`, `text` | Defines how the source is located and parsed; it does not select the objective or record schema. |

The record schema `type: text` and loader `loader.type: text` are distinct. The record schema requires a `text` field; the loader parses a remote text file into one `text` record per line.

Exactly one dataset is supported in this v1alpha1 contract. Its record schema must match the selected objective:

| Objective | Training signal | Allowed dataset record type | Defaults | Restrictions |
| --- | --- | --- | --- | --- |
| omitted, YAML `null`, empty mapping, or `sft` | Demonstrated outputs or complete sequences | Exactly one of `alpaca`, `messages`, `sharegpt`, `prompt-completion`, or `text` | `learningRate: 0.0002` | Rejects `preference`; existing SFT loss behavior is unchanged. |
| `dpo` | A `chosen` response preferred over a `rejected` response for one prompt | Exactly one `preference` dataset | `beta: 0.1`, `lossType: sigmoid`, `maxPromptLength: 512`, `learningRate: 0.000001` | Requires `packing: false`, rejects `config.unsloth.loss: response`, and requires `maxPromptLength <= maxSeqLength`. |

Default SFT serialization and cache keys omit the objective, so existing SFT configurations retain their prior training definition. A DPO `beta` must be finite and greater than zero. The initial API supports only `lossType: sigmoid`. DPO uses the LoRA policy with `ref_model=None`; the same PEFT model with its adapter disabled supplies reference log probabilities. This is not reference-free DPO and does not accept a user-selected reference model. Preference data bypasses every SFT formatter and reaches the DPO trainer as `prompt`, `chosen`, and `rejected` strings.

A complete DPO objective and dataset declaration is:

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

Each record has this explicit shape:

```json
{"prompt":"How should I rotate an API key?","chosen":"Deploy a replacement before revoking the old key.","rejected":"Revoke the old key before creating a replacement."}
```

### Dataset Loaders

The record `type`, training `objective`, and `loader.type` are independent. The record type defines required columns, the objective selects SFT or DPO, and the loader defines source transport and parsing. Loader and objective options are included in the serialized training configuration, so changing them invalidates training and downstream export without invalidating dependency installation.

| Loader | Source | Loader-specific fields | Reproducibility |
| --- | --- | --- | --- |
| omitted | Existing automatic behavior: HTTP(S) uses JSON; other sources go to Hugging Face Datasets | None | Mutable-source warning |
| `huggingface` | Hugging Face dataset identifier | Optional `subset`, `split`, `revision` | Warns when `revision` is omitted |
| `json` | Absolute HTTP(S) URL | Optional `split`, `checksum` | Warns when `checksum` is omitted |
| `csv` | Absolute HTTP(S) URL | Optional `split`, `checksum` | Warns when `checksum` is omitted |
| `parquet` | Absolute HTTP(S) URL | Optional `split`, `checksum` | Warns when `checksum` is omitted |
| `text` | Absolute HTTP(S) URL | Optional `split`, `checksum` | Warns when `checksum` is omitted |

`split` defaults to `train` and must contain letters, numbers, or underscores in one or more dot-separated segments. Hyphenated names and split expressions such as `train-sft` or `train[:10%]` are not supported by the pinned Datasets API. Split selection configures only the training dataset; evaluation datasets and metrics remain unsupported.

A Hugging Face `revision`, when present, must be a lowercase 40-character immutable commit hash. The Hugging Face loader rejects `checksum`. Remote-file loaders reject `subset` and `revision`; their optional checksum must use `sha256:<64 lowercase hex>` and covers the raw downloaded bytes before parsing or decompression. AIKit verifies both cached and newly downloaded bytes before invoking Hugging Face Datasets or allocating the model. Files are stored by content digest in an AIKit-owned namespace under `HF_DATASETS_CACHE`, with data and compression suffixes retained for parser detection.

Unknown fields, non-mapping values, nulls, and non-string values inside `loader` fail during parsing. Unknown fields elsewhere retain the existing permissive behavior. URL credentials, queries, and fragments are redacted from AIKit-generated warning logs and errors and are not written into cache filenames or cache metadata. The configured URL remains part of the serialized training configuration and BuildKit definition. This redaction therefore does not turn signed URLs into a supported secret channel; private dataset secret mounts are not supported.

The `text` file loader yields one `text` record per line and therefore cannot supply `preference` records. JSON, CSV, and Parquet files may contain any compatible record schema. For example:

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

```yaml
datasets:
  - source: https://datasets.example.com/train.parquet
    type: prompt-completion
    loader:
      type: parquet
      split: train
      checksum: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

### Dataset Types

Only one dataset is currently supported. Its `type` determines required columns and works with the objective compatibility rules above.

| Type | Required record shape | Loss behavior |
| --- | --- | --- |
| `alpaca` | `instruction`, `input`, `output` | Renders the existing Alpaca prompt, appends EOS, and supervises the full sequence. |
| `messages` | Non-empty `messages` list containing text-only `role` and `content` mappings | Applies the tokenizer's chat template. `loss: all` supervises the full rendered sequence; `loss: response` supervises assistant responses. |
| `sharegpt` | Non-empty `conversations` list containing string `from` and `value` mappings | Converts fixed ShareGPT roles to canonical messages, then uses the same rendering and loss pipeline as `messages`. |
| `prompt-completion` | `prompt`, non-empty `completion` | Keeps both columns separate, masks prompt tokens, and supervises completion and EOS tokens. |
| `text` | non-empty `text` | Preserves the preformatted sequence, normalizes BOS/EOS boundaries, and supervises the full sequence. |
| `preference` | Non-empty string `prompt`, `chosen`, and `rejected`; choices must differ | DPO only. Preserves all three strings, drops unrelated columns, and performs no SFT formatting or EOS appending. |

Empty datasets, missing or null fields, values of the wrong type, and unknown dataset types are rejected. Preference prompts and responses may not be empty or whitespace-only, and `chosen` must not equal `rejected`. Implicit prompt extraction and conversational preference arrays are not supported. Existing `alpaca` configurations retain their current rendering and full-sequence loss behavior.

AIKit projects exactly the three `preference` columns shown above before model allocation and does not append EOS text or run Alpaca, prompt-completion, text, messages, or ShareGPT preprocessing. Multiple preference datasets, implicit prompts, conversational preference arrays, evaluation datasets, alternate DPO losses, custom reference models, and reference-free DPO are outside the initial contract.

The chat loss setting defaults to `all` for backward compatibility when omitted or set to YAML `null`; an explicit empty string is invalid. `response` is accepted only for `messages` and `sharegpt`; `alpaca`, `prompt-completion`, and `text` retain their fixed loss behavior. Response-only training requires `packing: false` so masking cannot cross conversation boundaries. It derives markers from the model's deterministic chat template and uses Unsloth's response masking after trainer construction. AIKit rejects marker strings or token matches that collide with message content or fail to match the rendered role boundaries, then validates the actual prepared labels before training. If marker derivation fails, labels do not match the response spans, or a prepared dataset has no supervised response tokens, training fails without falling back to `all`. Native assistant-only loss, custom chat templates, custom markers, and per-dataset loss settings are not supported.

For example, a prompt-completion dataset entry and record are:

```yaml
datasets:
  - source: organization/question-answer-data
    type: prompt-completion
```

```json
{"prompt":"Question: What is a container image?\nAnswer:","completion":" An immutable package containing an application and its dependencies."}
```

For a `messages` dataset, each record is a canonical text-only conversation:

```yaml
datasets:
  - source: organization/chat-data
    type: messages
```

```json
{"messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"What is a container image?"},{"role":"assistant","content":"An immutable application package."}]}
```

The `messages` list must be non-empty. Benign top-level metadata columns, such as IDs and source labels, are ignored when records are projected to `messages`; top-level tool, template-control, document, image, audio, and video fields are rejected. Every turn must contain exactly the string fields `role` and `content`; only `system`, `user`, and `assistant` roles are supported. Each conversation must contain at least one assistant turn and end with an assistant turn. Extra fields within turns, tool calls, structured content, and multimodal content are rejected. AIKit does not accept a custom chat template.

The base model tokenizer must provide a usable, deterministic chat template. Wall-clock-dependent templates containing `strftime_now` are rejected until deterministic template values can be configured and included in cache keys. Before LoRA allocation, AIKit calls `apply_chat_template` with `tokenize=False` and `add_generation_prompt=False`, stores the result in the canonical `text` field, and sends rendered text rather than raw messages to the locked Unsloth SFT path. It adds no special tokens to that rendering. Direct chat-template token IDs must exactly match the locked rendered-text tokenization; mismatches and records over `maxSeqLength` fail instead of being silently changed or truncated. Source-aware errors redact URL credentials and query values.

Messages use full-sequence SFT when `loss` is omitted or set to `all`: system, user, assistant, and template tokens are supervised, and existing packing and role-order behavior is preserved. With `loss: response`, only assistant responses are supervised, `packing` must be `false`, any system messages must form a prefix, and the remaining messages must alternate user and assistant in complete pairs. AIKit validates unique marker placement and the actual prepared labels before training.

For a ShareGPT dataset, each record contains a text-only `conversations` list:

```yaml
datasets:
  - source: organization/sharegpt-data
    type: sharegpt
config:
  unsloth:
    loss: response
```

```json
{"conversations":[{"from":"system","value":"You are concise."},{"from":"human","value":"What is a container image?"},{"from":"gpt","value":"An immutable application package."}]}
```

Every turn must provide string `from` and `value` fields. The adapter maps `system` to `system`; `human` and `user` to `user`; and `gpt` and `assistant` to `assistant`. Unknown roles, missing fields, non-string content, and empty conversations are rejected. Alternate keys and role names are not inferred. AIKit projects `conversations` independently of benign top-level metadata, so a valid one-record dataset is accepted without heuristic schema detection.

The adapter produces canonical text-only messages and then uses the `messages` validation, deterministic chat-template rendering, token-equivalence, sequence-length, and loss checks. Packing is supported with `loss: all` and rejected with `loss: response`. The normalized conversation must contain an assistant response and end with an assistant turn. Tools, custom templates or markers, and structured or multimodal content are not supported.

For a `text` dataset, each record is already the complete sequence to train on:

```yaml
datasets:
  - source: organization/domain-corpus
    type: text
```

```json
{"text":"Question: What is a container image?\nAnswer: An immutable package containing an application and its dependencies."}
```

AIKit preserves the source content except for tokenizer-aware special-token boundary normalization. Tokenizing a normalized record yields exactly one effective leading BOS where applicable and exactly one terminal EOS. Tokenizers that insert BOS or EOS automatically, as well as tokenizers that define no BOS, are supported; a tokenizer without a usable EOS token fails. Records whose normalized token sequence exceeds `maxSeqLength` are rejected instead of truncated so the terminal EOS remains intact. Full-sequence labels supervise the entire normalized record, and packing uses its tokenized EOS boundary.

This is full-sequence SFT, not continued pretraining. The `text` type does not change the standard LoRA targets or optimizer configuration, train embedding or language-model-head parameters, or add an embedding-specific learning rate.

Example:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
baseModel: unsloth/mistral-7b-instruct-v0.2-bnb-4bit
datasets:
  - source: yahma/alpaca-cleaned
    type: alpaca
config:
  unsloth:
    packing: false
    maxSeqLength: 2048
    loadIn4bit: true
    batchSize: 2
    gradientAccumulationSteps: 4
    warmupSteps: 10
    maxSteps: 60
    learningRate: 0.0002
    loggingSteps: 1
    optimizer: adamw_8bit
    weightDecay: 0.01
    lrSchedulerType: linear
    seed: 42
output:
  quantize: q4_k_m
  name: model
```
