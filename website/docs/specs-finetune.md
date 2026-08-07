---
title: Fine Tuning API Specifications
---

## v1alpha1

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: # required. only v1alpha1 is supported at the moment
baseModel: # required. any base model from Huggingface. for unsloth, see for 4bit pre-quantized models: https://huggingface.co/unsloth
datasets:
  - source: # required. this can be a Huggingface dataset repo or a URL pointing to a JSON or JSON Lines file
    type: # required. can be "alpaca", "messages", "sharegpt", "prompt-completion", or "text"
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

### Dataset Types

Only one dataset is currently supported. Its `type` determines the required columns and loss behavior.

| Type | Required record shape | Loss behavior |
| --- | --- | --- |
| `alpaca` | `instruction`, `input`, `output` | Renders the existing Alpaca prompt, appends EOS, and supervises the full sequence. |
| `messages` | Non-empty `messages` list containing text-only `role` and `content` mappings | Applies the tokenizer's chat template. `loss: all` supervises the full rendered sequence; `loss: response` supervises assistant responses. |
| `sharegpt` | Non-empty `conversations` list containing string `from` and `value` mappings | Converts fixed ShareGPT roles to canonical messages, then uses the same rendering and loss pipeline as `messages`. |
| `prompt-completion` | `prompt`, non-empty `completion` | Keeps both columns separate, masks prompt tokens, and supervises completion and EOS tokens. |
| `text` | non-empty `text` | Preserves the preformatted sequence, normalizes BOS/EOS boundaries, and supervises the full sequence. |

Empty datasets, missing or null fields, values of the wrong type, and unknown dataset types are rejected. Existing `alpaca` configurations retain their current rendering and full-sequence loss behavior.

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
