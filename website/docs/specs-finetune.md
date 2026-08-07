---
title: Fine Tuning API Specifications
---

## v1alpha1

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: # required. only v1alpha1 is supported at the moment
baseModel: # required. any base model from Huggingface. for unsloth, see for 4bit pre-quantized models: https://huggingface.co/unsloth
datasets: # required. one or more entries, concatenated in configured order after normalization
  - source: # required. a Hugging Face dataset identifier or an absolute HTTP(S) URL
    type: # required record schema. can be "alpaca", "messages", "sharegpt", "prompt-completion", or "text"
    loader: # optional. omission preserves automatic legacy loading and the train split
      type: # required when loader is present. huggingface, json, csv, parquet, or text
      subset: # optional. Hugging Face loader only
      split: # optional. defaults to train; selects training data, not evaluation data
      revision: # optional. Hugging Face loader only; lowercase 40-character commit hash
      checksum: # optional. remote-file loaders only; sha256:<64 lowercase hex> of raw downloaded bytes
config:
  unsloth:
    loss: # optional. omitted or null defaults to all. response is supported only for messages and sharegpt
    packing: # optional. defaults to false. not supported with loss: response.
    maxSeqLength: # optional. defaults to 2048
    loadIn4bit: # optional. defaults to true
    batchSize: # optional. default to 2
    gradientAccumulationSteps: # optional. defaults to 4
    warmupSteps: # optional. defaults to 10
    maxSteps: # optional. defaults to 60
    learningRate: # optional. defaults to 0.0002
    loggingSteps: # optional. defaults to 1
    optimizer: # optional. defaults to adamw_8bit
    weightDecay: # optional. defaults to 0.01
    lrSchedulerType: # optional. defaults to linear
    seed: # optional. defaults to 42
output:
  quantize: # optional. defaults to q4_k_m. for unsloth, see for allowed quantization methods: https://github.com/unslothai/unsloth/wiki#saving-to-gguf.
  name: # optional. defaults to "aikit-model"
```

### Dataset Format Model

Multiple-dataset composition in this configuration applies to supervised fine-tuning (SFT) only. It does not define a preference record schema, objective selection, or DPO training.

Each `datasets` entry has two independent format dimensions:

- `type` selects the record schema, required fields, and SFT supervision behavior.
- `loader.type` selects source transport and parsing.

A loader does not select or infer the record schema. The loaded columns must satisfy the configured record `type`.

| Record `type` | Required record shape | SFT supervision | Composition group |
| --- | --- | --- | --- |
| `alpaca` | String `instruction`, `input`, and `output` fields | Renders the Alpaca prompt, appends EOS, and supervises the full sequence | Full-sequence |
| `messages` | Non-empty `messages` list containing text-only `role` and `content` mappings | `loss: all` supervises the rendered sequence; `loss: response` supervises assistant responses | Full-sequence or response-only chat, according to global `loss` |
| `sharegpt` | Non-empty `conversations` list containing string `from` and `value` mappings | Normalizes fixed ShareGPT roles to messages and applies the same chat loss behavior | Full-sequence or response-only chat, according to global `loss` |
| `prompt-completion` | String `prompt` and non-empty string `completion` fields | Masks prompt tokens and supervises completion and EOS tokens | Completion-only |
| `text` | Non-empty string `text` field | Normalizes BOS/EOS boundaries and supervises the full sequence | Full-sequence |

`config.unsloth.loss` is global to the training job. It cannot vary by dataset entry.

### Dataset Loaders

The record `type` and `loader.type` are independent. Every dataset entry and its loader options are included in the serialized training configuration, so changing or reordering any entry, or changing its `type`, `subset`, `split`, `revision`, or `checksum`, invalidates training and downstream export without invalidating dependency installation.

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

The `text` file loader yields one record with a `text` field per line and is normally paired with record `type: text`. JSON, CSV, and Parquet files may contain any supported record schema whose required fields are present. For example:

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

### Multiple Dataset Composition

One or more datasets are supported for SFT. The global `config.unsloth.loss` and every dataset type must resolve to one compatibility group:

| Compatibility group | Supported entries | Training behavior |
| --- | --- | --- |
| Full-sequence | Any mix of `alpaca`, `text`, `messages`, and `sharegpt` with `loss: all` and compatible tokenizer special-token boundaries | Normalizes to a canonical string `text` column and supervises the full sequence |
| Completion-only | One or more `prompt-completion` entries | Preserves canonical string `prompt` and `completion` columns and supervises completion and EOS tokens |
| Response-only chat | Any mix of `messages` and `sharegpt` with `loss: response` | Normalizes to canonical rendered `text`, masks non-assistant tokens, and requires `packing: false` |

Entries from different groups cannot be combined. In particular, prompt-completion data cannot be mixed with full-sequence data, and response-only chat cannot be mixed with `alpaca`, `prompt-completion`, or `text`. Per-dataset loss, source weights, random interleaving, deduplication, and automatic over- or undersampling are not supported.

AIKit loads entries sequentially in YAML order, validates and normalizes every source independently, then concatenates their canonical records in configured order while preserving row order within each source. This ordering is the input to trainer preprocessing; packing and trainer-level sampling may reorder records afterward. A single entry bypasses concatenation and retains the established single-source path. Repeating an entry intentionally repeats its rows, and every source must contain at least one record.

The locked Unsloth SFT path derives one rendered-text `add_special_tokens` value from the first canonical record. Before concatenation, AIKit compares every full-sequence record's token IDs under that dataset-wide value with its token IDs under the source-specific value. A mix in which one source already renders BOS and another relies on tokenizer-added BOS is rejected with the failing `datasets[n]` index when the values change the effective tokens; otherwise dataset order could duplicate BOS on one source or omit it from another. No-op setting differences remain valid, and reordering cannot bypass a real boundary mismatch.

For multiple `prompt-completion` entries, AIKit also compares each source's effective token IDs and completion mask under its source-specific `add_special_tokens` value with the dataset-wide value selected from the first source. A mismatch that changes retained tokens or the prompt/completion boundary is rejected before concatenation; no-op value differences remain valid.

The following SFT configuration combines two full-sequence schemas with independent pinned loaders:

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

Empty sources, missing or null fields, values of the wrong type, incompatible semantic groups, and unknown dataset types are rejected with the failing `datasets[n]` index. Existing single-source `alpaca` and other supported configurations retain their current rendering and loss behavior. URL details remain redacted in indexed errors.

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
