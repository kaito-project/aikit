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
    type: # required. can be "alpaca", "prompt-completion", or "text"
config:
  unsloth:
    packing: # optional. defaults to false. can make training 5x faster for short sequences.
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

| Type | Required string columns | Loss behavior |
| --- | --- | --- |
| `alpaca` | `instruction`, `input`, `output` | Renders the existing Alpaca prompt, appends EOS, and supervises the full sequence. |
| `prompt-completion` | `prompt`, non-empty `completion` | Keeps both columns separate, masks prompt tokens, and supervises completion and EOS tokens. |
| `text` | non-empty `text` | Preserves the preformatted sequence, normalizes BOS/EOS boundaries, and supervises the full sequence. |

Empty datasets, missing or null fields, values of the wrong type, and unknown dataset types are rejected. Existing `alpaca` configurations retain their current rendering and full-sequence loss behavior.

For example, a prompt-completion dataset entry and record are:

```yaml
datasets:
  - source: organization/question-answer-data
    type: prompt-completion
```

```json
{"prompt":"Question: What is a container image?\nAnswer:","completion":" An immutable package containing an application and its dependencies."}
```

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
