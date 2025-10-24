---
title: Packaging Models as OCI Artifacts
---

:::caution[Experimental Feature]

This feature might change without preserving backwards compatibility.

:::

AIKit can package large language models into Open Container Initiative (OCI) artifacts. This enables distribution of models through any OCI‑compliant registry.

## Overview

AIKit provides an extensible OCI packaging system. At this time, two explicit build targets are provided:

- `packager/modelpack` – produces OCI artifacts that are compliant with CNCF sandbox project [ModelPack](https://github.com/modelpack/model-spec) specifications.

- `packager/generic` – produces generic OCI artifacts.

### Sources Supported

Specify the source with `--build-arg source=`:

- Local context directory (default if omitted)
- Subdirectory of context: `subdir/`
- Single local file
- Remote `HTTP`/`HTTPS` file URL
- Hugging Face model: `huggingface://<org>/<repo>` optionally with revision `@<rev>`

## Modelpack Target (`packager/modelpack`)

Command example:

```shell
docker buildx build \
  --build-arg BUILDKIT_SYNTAX=ghcr.io/kaito-project/aikit/aikit:latest \
  --target packager/modelpack \
  --build-arg source=huggingface://Qwen/Qwen3-0.6B \
  --build-arg name=qwen3 \
  --output=qwen -<<<""
```

Due to current BuildKit limitations, we can not push directly to a remote registry at this time. You must first output to a local OCI layout, then use a tool like [`skopeo`](https://github.com/containers/skopeo) or [`oras`](https://github.com/oras-project/oras) to copy the image to a remote registry.

```shell
export REGISTRY=docker.io/youruser/qwen3:0.6b
skopeo copy oci:qwen/layout docker://$REGISTRY
```

### Layer Categorization

Files are deterministically classified into lists:

- weights: `*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, `*.ckpt`, plus any other unknown file that's larger than 10MiB
- config: tokenizer/config JSON & small text/json defaults
- docs: readme/license/markdown
- code: `*.py`, `*.sh`, `*.ipynb`, `*.go`, `*.js`, `*.ts`
- dataset: `*.csv`, `*.tsv`, `*.jsonl`, `*.parquet`, `*.arrow`, `*.h5`, `*.npz`

Each category forms one or more layers depending on packaging mode (see below). Metadata (file path, size, optional bundle counts) is embedded as JSON annotations per layer.

### Packaging Modes (`--build-arg layer_packaging=`)

- `raw` – every file becomes an individual layer
- `tar` – categories (except weights) are aggregated into a tar; weights individually tarred
- `tar+gzip` – same as tar but gzip compressed
- `tar+zstd` – same as tar but zstd compressed

### Media Types & Specification

AIKit's Modelpack target implements the CNCF sandbox project [ModelPack specification](https://github.com/modelpack/model-spec/blob/main/docs/spec.md).

## Generic Target (`packager/generic`)

General purpose packaging for arbitrary files.

```bash
docker buildx build \
  --build-arg BUILDKIT_SYNTAX=ghcr.io/kaito-project/aikit/aikit:latest \
  --target packager/generic \
  --build-arg source=https://example.com/model.bin \
  --build-arg name=example \
  --build-arg layer_packaging=raw \
  --output=example -<<<""
```

then use a tool like [`skopeo`](https://github.com/containers/skopeo) or [`oras`](https://github.com/oras-project/oras) to copy the image to a remote registry.

```shell
export REGISTRY=docker.io/youruser/qwen3:0.6b
skopeo copy oci:qwen/layout docker://$REGISTRY
```

### Output Modes

`--build-arg generic_output_mode=files` produces a direct copy of the resolved source tree (no layout transformation). Otherwise the generic script builds an OCI layout with either per‑file (`raw`) or single aggregated archive layer (`tar`, `tar+gzip`, `tar+zstd`).

### Media Types (Generic)

- Raw mode now assigns layer media type: `application/octet-stream`
- Tar / compressed modes: standard image layer media type (`application/vnd.oci.image.layer.v1.tar`, `application/vnd.oci.image.layer.v1.tar+gzip`, `application/vnd.oci.image.layer.v1.tar+zstd`)

## Hugging Face Token Usage Example

You can provide a Hugging Face token for private model access using [Docker build secrets](https://docs.docker.com/build/building/secrets/).

```shell
export HF_TOKEN=<your_huggingface_token>

docker buildx build \
  --secret id=hf-token,env=HF_TOKEN \
  --build-arg BUILDKIT_SYNTAX=ghcr.io/kaito-project/aikit/aikit:latest \
  --target packager/modelpack \
  --build-arg source=huggingface://meta-llama/Llama-3.2-1B \
  --build-arg name=llama \
  --build-arg exclude="'original/**'" \
  --output=llama -<<<""
```

### Download exclusions (`--build-arg exclude=`)

When downloading from Hugging Face, you can specify files or directories to exclude using the `--build-arg exclude=` option. This is useful for omitting unnecessary files from the download process. Exclusions use glob patterns and should be provided as a single string with space-separated patterns.

For example, to exclude the `original` and `metal` directories, you can use the following command:

```shell
--build-arg exclude="'original/*' 'metal/*'"
```
