#!/usr/bin/env python3

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import math
import operator
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

TRAIN_CONFIG_PATH = Path("/aikit-config/train-config.yaml")
EXPORT_CONFIG_PATH = Path("/aikit-config/export-config.yaml")
TRAINED_MODEL_DIRECTORY = Path("/aikit-trained-model")
EXPORT_DIRECTORY = Path("/aikit-unsloth-export")
ARTIFACT_DIRECTORY = Path("/model")
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
DEFAULT_ADAPTER_NAME = "default"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
HF_COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATASET_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
DATASET_SPLIT_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
GO_YAML_SCIENTIFIC_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+$"
)
DATASET_LOADER_HUGGINGFACE = "huggingface"
DATASET_LOADER_JSON = "json"
DATASET_LOADER_CSV = "csv"
DATASET_LOADER_PARQUET = "parquet"
DATASET_LOADER_TEXT = "text"
SUPPORTED_DATASET_LOADERS = frozenset(
    (
        DATASET_LOADER_HUGGINGFACE,
        DATASET_LOADER_JSON,
        DATASET_LOADER_CSV,
        DATASET_LOADER_PARQUET,
        DATASET_LOADER_TEXT,
    )
)
REMOTE_DATASET_LOADERS = frozenset(
    (
        DATASET_LOADER_JSON,
        DATASET_LOADER_CSV,
        DATASET_LOADER_PARQUET,
        DATASET_LOADER_TEXT,
    )
)
DEFAULT_DATASET_SPLIT = "train"
REMOTE_DATASET_CACHE_SUBDIRECTORY = "aikit-remote-files"
REMOTE_DATASET_DEFAULT_SUFFIX = {
    DATASET_LOADER_JSON: ".json",
    DATASET_LOADER_CSV: ".csv",
    DATASET_LOADER_PARQUET: ".parquet",
    DATASET_LOADER_TEXT: ".txt",
}
REMOTE_DATASET_DATA_SUFFIXES = {
    DATASET_LOADER_JSON: frozenset((".json", ".jsonl", ".ndjson")),
    DATASET_LOADER_CSV: frozenset((".csv",)),
    DATASET_LOADER_PARQUET: frozenset((".parquet",)),
    DATASET_LOADER_TEXT: frozenset((".txt", ".text")),
}
REMOTE_DATASET_COMPRESSION_SUFFIXES = frozenset(
    (".bz2", ".gz", ".xz", ".zip", ".zst")
)
REMOTE_DATASET_CHUNK_SIZE = 1024 * 1024
# Bound both connection establishment and each blocking response read.
REMOTE_DATASET_REQUEST_TIMEOUT_SECONDS = 60.0
# Bound aggregate download time even when a server keeps each read active.
REMOTE_DATASET_TOTAL_TIMEOUT_SECONDS = 60.0 * 60.0
# Prevent one remote file from consuming unbounded temporary and cache storage.
REMOTE_DATASET_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024
DATASET_TYPE_ALPACA = "alpaca"
DATASET_TYPE_MESSAGES = "messages"
DATASET_TYPE_PREFERENCE = "preference"
DATASET_TYPE_PROMPT_COMPLETION = "prompt-completion"
DATASET_TYPE_SHAREGPT = "sharegpt"
DATASET_TYPE_TEXT = "text"
SUPPORTED_DATASET_TYPES = frozenset(
    (
        DATASET_TYPE_ALPACA,
        DATASET_TYPE_MESSAGES,
        DATASET_TYPE_PREFERENCE,
        DATASET_TYPE_PROMPT_COMPLETION,
        DATASET_TYPE_SHAREGPT,
        DATASET_TYPE_TEXT,
    )
)
DATASET_REQUIRED_FIELDS = {
    DATASET_TYPE_ALPACA: ("instruction", "input", "output"),
    DATASET_TYPE_MESSAGES: ("messages",),
    DATASET_TYPE_PREFERENCE: ("prompt", "chosen", "rejected"),
    DATASET_TYPE_PROMPT_COMPLETION: ("prompt", "completion"),
    DATASET_TYPE_SHAREGPT: ("conversations",),
    DATASET_TYPE_TEXT: ("text",),
}
MESSAGE_FIELDS = frozenset(("role", "content"))
SUPPORTED_MESSAGE_ROLES = frozenset(("system", "user", "assistant"))
SHAREGPT_MESSAGE_FIELDS = frozenset(("from", "value"))
SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}
CHAT_DATASET_TYPES = frozenset((DATASET_TYPE_MESSAGES, DATASET_TYPE_SHAREGPT))
LOSS_ALL = "all"
LOSS_RESPONSE = "response"
SUPPORTED_LOSSES = frozenset((LOSS_ALL, LOSS_RESPONSE))
OBJECTIVE_TYPE_SFT = "sft"
OBJECTIVE_TYPE_DPO = "dpo"
SUPPORTED_OBJECTIVE_TYPES = frozenset((OBJECTIVE_TYPE_SFT, OBJECTIVE_TYPE_DPO))
DPO_LOSS_SIGMOID = "sigmoid"
DPO_TRUNCATION_KEEP_END = "keep_end"
DEFAULT_DPO_BETA = 0.1
DEFAULT_DPO_MAX_PROMPT_LENGTH = 512
DATASET_COMPATIBILITY_FULL_SEQUENCE = "full-sequence"
DATASET_COMPATIBILITY_PROMPT_COMPLETION = "completion-only"
DATASET_COMPATIBILITY_RESPONSE_CHAT = "response-only chat"
UNSUPPORTED_MESSAGES_TOP_LEVEL_FIELDS = frozenset(
    (
        "add_generation_prompt",
        "audio",
        "audio_path",
        "audio_paths",
        "audio_url",
        "audio_urls",
        "audios",
        "chat_template",
        "chat_template_kwargs",
        "continue_final_message",
        "documents",
        "function_call",
        "functions",
        "image",
        "image_path",
        "image_paths",
        "image_url",
        "image_urls",
        "images",
        "tool",
        "tool_calls",
        "tool_choice",
        "tools",
        "tokenizer_kwargs",
        "video",
        "video_path",
        "video_paths",
        "video_url",
        "video_urls",
        "videos",
    )
)
PREPROCESSING_VERIFICATION_PROMPT = "Question: What is 2 + 2?\nAnswer:"
PREPROCESSING_VERIFICATION_COMPLETION = " 4."
PROMPT_COMPLETION_PREPROCESSING_VERIFICATION_MAX_LENGTH = 4096
PROMPT_COMPLETION_VALIDATION_BATCH_SIZE = 128
# Bound the estimated retained token IDs for the prompt and combined text.
PROMPT_COMPLETION_VALIDATION_TOKEN_BUDGET = 262_144
PROMPT_PREFIX_FINGERPRINT_MASK = (1 << 256) - 1
TEXT_VALIDATION_BATCH_SIZE = 128
# Bound the retained token IDs across each text validation batch.
TEXT_VALIDATION_TOKEN_BUDGET = 262_144
TEXT_PREPROCESSING_VERIFICATION_TEXTS = (
    "AIKit text boundary verification one.",
    "AIKit text boundary verification two!",
)
MESSAGES_VALIDATION_BATCH_SIZE = 128
# Bound canonical and rendered-path token IDs retained for one validation batch.
MESSAGES_VALIDATION_TOKEN_BUDGET = 262_144
MESSAGES_TOKEN_FINGERPRINT_MASK = (1 << 256) - 1

# Keep the Alpaca prompt byte-for-byte compatible with existing fine-tuning builds.
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


class DatasetSourceKind(Enum):
    JSON_URL = "json_url"
    DATASET = "dataset"


class DatasetLoadSpec(NamedTuple):
    path: str
    kwargs: dict[str, Any]


class DatasetLoaderSpec(NamedTuple):
    loader_type: str | None
    subset: str | None
    split: str
    revision: str | None
    checksum: str | None


class TrainingDatasetSpec(NamedTuple):
    source: str
    dataset_type: str
    loader: DatasetLoaderSpec
    index: int | None = None


class TrainingObjectiveSpec(NamedTuple):
    objective_type: str
    beta: float | None
    loss_type: str | None
    max_prompt_length: int | None


class PromptPrefixFingerprint(NamedTuple):
    sequence_count: int
    first_digest_sum: int
    second_digest_sum: int


class MessagesTokenFingerprint(NamedTuple):
    sequence_count: int
    first_digest_sum: int
    second_digest_sum: int


class ResponseMarkers(NamedTuple):
    instruction_part: str
    response_part: str
    instruction_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    use_tokenizer_parts: bool


class MessagesRenderError(RuntimeError):
    pass


class ShareGPTNormalizationError(RuntimeError):
    pass


class TextBoundaryPolicy(NamedTuple):
    eos_token: str
    eos_token_id: int
    bos_token: str | None
    bos_token_id: int | None
    add_special_tokens: bool
    append_eos_token: bool


class TrainDependencies(NamedTuple):
    fast_language_model: Any
    is_bfloat16_supported: Callable[[], bool]
    dataset_from_dict: Callable[..., Any]
    load_dataset: Callable[..., Any]
    model_info: Callable[..., Any]
    resolve_model_name: Callable[..., str]
    sft_config: Callable[..., Any]
    sft_trainer: Callable[..., Any]
    dpo_config: Callable[..., Any]
    dpo_trainer: Callable[..., Any]
    get_chat_template_parts: Callable[..., tuple[str, str]]
    train_on_responses_only: Callable[..., Any]
    concatenate_datasets: Callable[[list[Any]], Any] | None = None


class ExportDependencies(NamedTuple):
    fast_language_model: Any
    snapshot_download: Callable[..., str]


def parse_config(
    config_text: str,
    *,
    loader: Callable[[str], Any] | None = None,
) -> Mapping[str, Any]:
    if loader is None:
        import yaml

        loader = yaml.safe_load

    config = loader(config_text)
    if not isinstance(config, Mapping):
        raise ValueError(
            f"configuration root must be a mapping, got {type(config).__name__}"
        )

    return config


def load_config(
    config_path: Path | str,
    *,
    loader: Callable[[str], Any] | None = None,
) -> Mapping[str, Any]:
    config_text = Path(config_path).read_text(encoding="utf-8")
    return parse_config(config_text, loader=loader)


def unsloth_config(train_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return train_config["config"]["unsloth"]


def output_config(export_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return export_config["output"]


def normalize_go_yaml_float(
    value: Any,
    *,
    description: str,
    allow_zero: bool,
) -> float:
    """Normalize Go YAML scientific scalars and enforce their numeric range."""
    requirement = "zero or greater" if allow_zero else "greater than zero"
    error_message = f"{description} must be a finite value {requirement}"

    if isinstance(value, str):
        if GO_YAML_SCIENTIFIC_FLOAT_PATTERN.fullmatch(value) is None:
            raise ValueError(error_message)
        value = float(value)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(error_message)

    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or normalized < 0
        or (not allow_zero and normalized == 0)
    ):
        raise ValueError(error_message)

    return normalized


def training_objective_spec(
    train_config: Mapping[str, Any],
) -> TrainingObjectiveSpec:
    objective = train_config.get("objective")
    if objective is None:
        objective = {}
    if not isinstance(objective, Mapping):
        raise ValueError("training objective must be a mapping")
    if any(not isinstance(field, str) for field in objective):
        raise ValueError("training objective field names must be strings")

    allowed_fields = frozenset(
        ("type", "beta", "lossType", "maxPromptLength")
    )
    unknown_fields = sorted(set(objective) - allowed_fields)
    if unknown_fields:
        quoted_fields = ", ".join(repr(field) for field in unknown_fields)
        raise ValueError(
            f"training objective contains unknown fields: {quoted_fields}"
        )

    configured_type = objective.get("type", OBJECTIVE_TYPE_SFT)
    objective_type = (
        OBJECTIVE_TYPE_SFT if configured_type is None else configured_type
    )
    if (
        not isinstance(objective_type, str)
        or objective_type not in SUPPORTED_OBJECTIVE_TYPES
    ):
        raise ValueError(f"unsupported training objective {objective_type!r}")

    if objective_type == OBJECTIVE_TYPE_SFT:
        dpo_fields = sorted(
            field
            for field in ("beta", "lossType", "maxPromptLength")
            if field in objective
        )
        if dpo_fields:
            quoted_fields = ", ".join(repr(field) for field in dpo_fields)
            raise ValueError(
                "SFT objective does not support DPO fields: "
                f"{quoted_fields}"
            )
        return TrainingObjectiveSpec(
            objective_type=OBJECTIVE_TYPE_SFT,
            beta=None,
            loss_type=None,
            max_prompt_length=None,
        )

    configured_beta = objective.get("beta", DEFAULT_DPO_BETA)
    beta = normalize_go_yaml_float(
        DEFAULT_DPO_BETA if configured_beta is None else configured_beta,
        description="DPO objective beta",
        allow_zero=False,
    )

    configured_loss_type = objective.get("lossType", DPO_LOSS_SIGMOID)
    loss_type = (
        DPO_LOSS_SIGMOID
        if configured_loss_type is None
        else configured_loss_type
    )
    if not isinstance(loss_type, str) or loss_type != DPO_LOSS_SIGMOID:
        raise ValueError(f"unsupported DPO objective loss type {loss_type!r}")

    configured_max_prompt_length = objective.get(
        "maxPromptLength",
        DEFAULT_DPO_MAX_PROMPT_LENGTH,
    )
    max_prompt_length = (
        DEFAULT_DPO_MAX_PROMPT_LENGTH
        if configured_max_prompt_length is None
        else configured_max_prompt_length
    )
    if (
        isinstance(max_prompt_length, bool)
        or not isinstance(max_prompt_length, int)
        or max_prompt_length <= 0
    ):
        raise ValueError(
            "DPO objective maxPromptLength must be an integer greater than zero"
        )

    return TrainingObjectiveSpec(
        objective_type=OBJECTIVE_TYPE_DPO,
        beta=beta,
        loss_type=loss_type,
        max_prompt_length=max_prompt_length,
    )


def require_hf_commit_hash(revision: Any, *, description: str) -> str:
    if (
        not isinstance(revision, str)
        or HF_COMMIT_HASH_PATTERN.fullmatch(revision) is None
    ):
        raise RuntimeError(
            f"{description} is not an immutable Hugging Face commit hash"
        )

    return revision


def resolve_model_snapshot(
    configured_model_name: str,
    *,
    load_in_4bit: bool,
    description: str,
    model_info: Callable[..., Any],
    resolve_model_name: Callable[..., str],
) -> tuple[str, str]:
    resolved_model_name = resolve_model_name(
        configured_model_name,
        load_in_4bit=load_in_4bit,
    )
    revision = getattr(model_info(repo_id=resolved_model_name), "sha", None)

    return resolved_model_name, require_hf_commit_hash(
        revision,
        description=description,
    )


def resolve_export_base_model(
    configured_model_name: str,
    *,
    model_info: Callable[..., Any],
    resolve_model_name: Callable[..., str],
) -> tuple[str, str]:
    return resolve_model_snapshot(
        configured_model_name,
        load_in_4bit=False,
        description="resolved export base model revision",
        model_info=model_info,
        resolve_model_name=resolve_model_name,
    )


def pin_peft_base_model(
    model: Any,
    *,
    base_model_name: str,
    revision: str,
) -> None:
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, Mapping) or not peft_configs:
        raise RuntimeError("trained model does not expose a PEFT configuration")

    for peft_config in peft_configs.values():
        peft_config.base_model_name_or_path = base_model_name
        peft_config.revision = revision


def validate_adapter_save_contract(model: Any) -> None:
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, Mapping) or set(peft_configs) != {
        DEFAULT_ADAPTER_NAME
    }:
        raise RuntimeError(
            "trained model must expose exactly one PEFT adapter named default"
        )

    peft_config = peft_configs[DEFAULT_ADAPTER_NAME]
    for field in ("modules_to_save", "trainable_token_indices"):
        if getattr(peft_config, field, None):
            raise RuntimeError(
                f"trained adapter uses unsupported PEFT {field} state"
            )
    if getattr(model, "_need_to_train_embeddings", False) is True:
        raise RuntimeError(
            "trained adapter unexpectedly requires embedding layers"
        )


def validate_portable_adapter_bundle(
    trained_model_directory: Path | str,
) -> None:
    trained_model_path = Path(trained_model_directory)
    required_files = (
        ADAPTER_CONFIG_FILENAME,
        ADAPTER_WEIGHTS_FILENAME,
        TOKENIZER_CONFIG_FILENAME,
    )
    for filename in required_files:
        artifact_path = trained_model_path / filename
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or artifact_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"saved adapter is missing a non-empty {filename}"
            )

    for artifact_path in trained_model_path.rglob("*"):
        if artifact_path.is_symlink():
            raise RuntimeError(
                f"saved adapter contains a symbolic link: {artifact_path}"
            )

        relative_path = artifact_path.relative_to(trained_model_path)
        if artifact_path.is_dir():
            continue
        if not artifact_path.is_file():
            raise RuntimeError(
                f"saved adapter contains a special file: {relative_path}"
            )
        filename = artifact_path.name
        if (
            filename == "adapter_model.bin"
            or filename == "config.json"
            or filename.endswith(".gguf")
            or filename.startswith("pytorch_model")
            or (
                filename.startswith("model")
                and filename.endswith(".safetensors")
            )
            or (
                filename == ADAPTER_CONFIG_FILENAME
                and relative_path.parent != Path(".")
            )
        ):
            raise RuntimeError(
                f"saved adapter contains unsupported artifact: {relative_path}"
            )

    adapter_config = dict(
        load_config(
            trained_model_path / ADAPTER_CONFIG_FILENAME,
            loader=json.loads,
        )
    )
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise RuntimeError("saved adapter is not a PEFT LoRA adapter")

    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise RuntimeError("saved adapter does not identify its base model")
    if (
        Path(base_model).is_absolute()
        or base_model.startswith((".", "~"))
        or "\\" in base_model
    ):
        raise RuntimeError("saved adapter base model must be a portable Hub ID")

    require_hf_commit_hash(
        adapter_config.get("revision"),
        description="saved adapter base model revision",
    )


def pin_adapter_base_model_snapshot(
    trained_model_directory: Path | str,
    *,
    snapshot_download: Callable[..., str],
) -> Path:
    adapter_config_path = Path(trained_model_directory) / ADAPTER_CONFIG_FILENAME
    adapter_config = dict(load_config(adapter_config_path, loader=json.loads))

    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise RuntimeError("saved adapter does not identify its base model")

    revision = require_hf_commit_hash(
        adapter_config.get("revision"),
        description="saved adapter base model revision",
    )
    snapshot_path = Path(
        snapshot_download(repo_id=base_model, revision=revision)
    )

    # Pinned Unsloth ignores PEFT's base-model revision, but correctly loads
    # an immutable local snapshot.
    adapter_config["base_model_name_or_path"] = str(snapshot_path)
    adapter_config_path.write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def has_http_dataset_scheme(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def is_http_dataset_source(source: str) -> bool:
    try:
        parsed_source = urlparse(source)
    except ValueError:
        return False
    return (
        parsed_source.scheme.lower() in {"http", "https"}
        and bool(parsed_source.netloc)
    )


def classify_dataset_source(source: str) -> DatasetSourceKind:
    if is_http_dataset_source(source):
        return DatasetSourceKind.JSON_URL

    return DatasetSourceKind.DATASET


def parse_dataset_loader_spec(
    dataset: Mapping[str, Any],
    *,
    source: str,
    dataset_index: int | None = None,
) -> DatasetLoaderSpec:
    path = (
        "training dataset loader"
        if dataset_index is None
        else f"datasets[{dataset_index}].loader"
    )
    if "loader" not in dataset:
        return DatasetLoaderSpec(
            loader_type=None,
            subset=None,
            split=DEFAULT_DATASET_SPLIT,
            revision=None,
            checksum=None,
        )

    loader = dataset["loader"]
    if not isinstance(loader, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if any(not isinstance(field, str) for field in loader):
        raise ValueError(f"{path} field names must be strings")

    allowed_fields = frozenset(
        ("type", "subset", "split", "revision", "checksum")
    )
    unknown_fields = sorted(set(loader) - allowed_fields)
    if unknown_fields:
        quoted_fields = ", ".join(repr(field) for field in unknown_fields)
        raise ValueError(
            f"{path} contains unknown fields: {quoted_fields}"
        )

    for field, value in loader.items():
        if not isinstance(value, str):
            raise ValueError(
                f"{path} field {field!r} must be a string"
            )
        if field in {"subset", "revision", "checksum"} and not value.strip():
            raise ValueError(
                f"{path} field {field!r} must not be empty"
            )

    loader_type = loader.get("type")
    if not isinstance(loader_type, str) or not loader_type.strip():
        raise ValueError(f"{path} type must be a non-empty string")
    if loader_type not in SUPPORTED_DATASET_LOADERS:
        raise ValueError(
            f"{path}: unsupported training dataset loader {loader_type!r}"
        )

    split = loader.get("split", DEFAULT_DATASET_SPLIT)
    if not isinstance(split, str) or not DATASET_SPLIT_PATTERN.fullmatch(split):
        raise ValueError(
            f"{path} split must be a named split containing "
            "letters, numbers, or underscores in dot-separated segments"
        )

    subset = loader.get("subset")
    revision = loader.get("revision")
    checksum = loader.get("checksum")
    if loader_type == DATASET_LOADER_HUGGINGFACE:
        if is_http_dataset_source(source):
            raise ValueError(
                f"{path}: huggingface dataset loader does not support an HTTP(S) source"
            )
        if checksum is not None:
            raise ValueError(
                f"{path}: huggingface dataset loader does not support checksum"
            )
        if revision is not None and not HF_COMMIT_HASH_PATTERN.fullmatch(
            revision
        ):
            raise ValueError(
                f"{path} revision must be a lowercase "
                "40-character commit hash"
            )
    else:
        if not is_http_dataset_source(source):
            raise ValueError(
                f"{path}: {loader_type} dataset loader requires an absolute HTTP(S) "
                "source"
            )
        if subset is not None:
            raise ValueError(
                f"{path}: remote-file dataset loaders do not support subset"
            )
        if revision is not None:
            raise ValueError(
                f"{path}: remote-file dataset loaders do not support revision"
            )
        if checksum is not None and not DATASET_CHECKSUM_PATTERN.fullmatch(
            checksum
        ):
            raise ValueError(
                f"{path}: remote-file dataset loader checksum must use lowercase "
                "sha256:<64 hex> format"
            )

    return DatasetLoaderSpec(
        loader_type=loader_type,
        subset=subset,
        split=split,
        revision=revision,
        checksum=checksum,
    )


def dataset_load_spec(
    dataset_spec: TrainingDatasetSpec,
    *,
    local_file: Path | None = None,
) -> DatasetLoadSpec:
    loader = dataset_spec.loader
    if loader.loader_type is None:
        if classify_dataset_source(dataset_spec.source) is DatasetSourceKind.JSON_URL:
            if local_file is None:
                raise ValueError(
                    "remote dataset must be materialized before loading"
                )
            return DatasetLoadSpec(
                path=DATASET_LOADER_JSON,
                kwargs={
                    "data_files": {DEFAULT_DATASET_SPLIT: str(local_file)},
                    "split": DEFAULT_DATASET_SPLIT,
                },
            )
        return DatasetLoadSpec(
            path=dataset_spec.source,
            kwargs={"split": DEFAULT_DATASET_SPLIT},
        )

    if loader.loader_type == DATASET_LOADER_HUGGINGFACE:
        kwargs: dict[str, Any] = {"split": loader.split}
        if loader.subset is not None:
            kwargs["name"] = loader.subset
        if loader.revision is not None:
            kwargs["revision"] = loader.revision
        return DatasetLoadSpec(path=dataset_spec.source, kwargs=kwargs)

    if local_file is None:
        raise ValueError("remote dataset must be materialized before loading")
    return DatasetLoadSpec(
        path=loader.loader_type,
        kwargs={
            "data_files": {loader.split: str(local_file)},
            "split": loader.split,
        },
    )


def dataset_source_description(
    source: str,
    *,
    loader_type: str | None = None,
) -> str:
    if is_http_dataset_source(source) or has_http_dataset_scheme(source):
        effective_loader = loader_type or DATASET_LOADER_JSON
        loader_label = {
            DATASET_LOADER_JSON: "JSON",
            DATASET_LOADER_CSV: "CSV",
            DATASET_LOADER_PARQUET: "Parquet",
            DATASET_LOADER_TEXT: "text",
        }.get(effective_loader, "remote")
        return f"remote {loader_label} URL"

    return f"source {source!r}"


def training_dataset_specs(
    train_config: Mapping[str, Any],
) -> tuple[TrainingDatasetSpec, ...]:
    datasets = train_config.get("datasets")
    if (
        not isinstance(datasets, Sequence)
        or isinstance(datasets, (str, bytes))
        or len(datasets) == 0
    ):
        raise ValueError("training configuration must define at least one dataset")

    specs = []
    for dataset_index, dataset in enumerate(datasets):
        path = f"datasets[{dataset_index}]"
        if not isinstance(dataset, Mapping):
            raise ValueError(f"{path} must be a mapping")

        dataset_type = dataset.get("type")
        if (
            not isinstance(dataset_type, str)
            or dataset_type not in SUPPORTED_DATASET_TYPES
        ):
            raise ValueError(
                f"{path}.type has unsupported dataset type {dataset_type!r}"
            )

        source = dataset.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}.source must be a non-empty string")
        if source != source.strip():
            raise ValueError(
                f"{path}.source must not have leading or trailing whitespace"
            )
        if has_http_dataset_scheme(source) and not is_http_dataset_source(source):
            raise ValueError(
                f"{path} HTTP(S) source must be an absolute URL with a host"
            )

        specs.append(
            TrainingDatasetSpec(
                source=source,
                dataset_type=dataset_type,
                loader=parse_dataset_loader_spec(
                    dataset,
                    source=source,
                    dataset_index=dataset_index,
                ),
                index=dataset_index,
            )
        )

    return tuple(specs)


def training_dataset_spec(
    train_config: Mapping[str, Any],
) -> TrainingDatasetSpec:
    specs = training_dataset_specs(train_config)
    if len(specs) != 1:
        raise ValueError("training configuration must define exactly one dataset")
    return specs[0]


def dataset_cache_directory() -> Path:
    configured_cache = os.environ.get("HF_DATASETS_CACHE")
    if configured_cache:
        base_cache = Path(configured_cache)
    else:
        base_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    return base_cache / REMOTE_DATASET_CACHE_SUBDIRECTORY


def remote_dataset_file_suffix(source: str, loader_type: str) -> str:
    suffixes = [suffix.lower() for suffix in Path(urlparse(source).path).suffixes]
    compression_suffix = ""
    if suffixes and suffixes[-1] in REMOTE_DATASET_COMPRESSION_SUFFIXES:
        compression_suffix = suffixes.pop()

    data_suffix = REMOTE_DATASET_DEFAULT_SUFFIX[loader_type]
    if suffixes and suffixes[-1] in REMOTE_DATASET_DATA_SUFFIXES[loader_type]:
        data_suffix = suffixes[-1]
    return data_suffix + compression_suffix


def nofollow_open_flags(flags: int) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError(
            "remote dataset cache requires no-follow file support"
        )
    return flags | nofollow | getattr(os, "O_CLOEXEC", 0)


@contextmanager
def open_dataset_cache_directory(cache_path: Path) -> Iterator[int]:
    cache_descriptor: int | None = None
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_descriptor = os.open(
            cache_path,
            nofollow_open_flags(
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            ),
        )
        if not stat.S_ISDIR(os.fstat(cache_descriptor).st_mode):
            raise OSError(errno.ENOTDIR, "cache path is not a directory")
    except (OSError, RuntimeError):
        if cache_descriptor is not None:
            os.close(cache_descriptor)
        raise RuntimeError(
            "remote dataset cache directory could not be opened safely"
        ) from None

    try:
        yield cache_descriptor
    finally:
        os.close(cache_descriptor)


@contextmanager
def dataset_digest_lock(
    cache_descriptor: int,
    digest: str,
) -> Iterator[None]:
    lock_name = f".{digest}.lock"
    lock_descriptor: int | None = None
    lock_acquired = False
    try:
        lock_descriptor = os.open(
            lock_name,
            nofollow_open_flags(os.O_RDWR | os.O_CREAT),
            0o600,
            dir_fd=cache_descriptor,
        )
        descriptor_stat = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
        ):
            raise OSError(errno.EINVAL, "cache lock is not a regular file")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        lock_acquired = True
        path_stat = os.stat(
            lock_name,
            dir_fd=cache_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise OSError(errno.EAGAIN, "cache lock path changed")
    except (OSError, RuntimeError):
        if lock_descriptor is not None:
            if lock_acquired:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        raise RuntimeError(
            "remote dataset cache lock could not be acquired safely"
        ) from None

    try:
        yield
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def open_regular_cache_entry(
    cache_descriptor: int,
    entry_name: str,
    max_file_bytes: int | None = None,
) -> int | None:
    try:
        entry_descriptor = os.open(
            entry_name,
            nofollow_open_flags(
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            ),
            dir_fd=cache_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            return None
        raise RuntimeError(
            "remote dataset cache entry could not be opened safely"
        ) from None

    entry_stat = os.fstat(entry_descriptor)
    if (
        not stat.S_ISREG(entry_stat.st_mode)
        or entry_stat.st_nlink != 1
        or (
            max_file_bytes is not None
            and entry_stat.st_size > max_file_bytes
        )
    ):
        os.close(entry_descriptor)
        return None
    return entry_descriptor


def write_file_descriptor(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "failed to write dataset file")
        remaining = remaining[written:]


def copy_file_descriptor(
    source_descriptor: int,
    target_descriptor: int,
    *,
    max_copy_bytes: int | None = None,
) -> str | None:
    original_offset = os.lseek(source_descriptor, 0, os.SEEK_CUR)
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while chunk := os.read(
            source_descriptor,
            REMOTE_DATASET_CHUNK_SIZE,
        ):
            copied_bytes += len(chunk)
            if max_copy_bytes is not None and copied_bytes > max_copy_bytes:
                return None
            write_file_descriptor(target_descriptor, chunk)
            digest.update(chunk)
    finally:
        os.lseek(source_descriptor, original_offset, os.SEEK_SET)
    return digest.hexdigest()


def create_verified_dataset_snapshot(
    cache_descriptor: int,
    entry_name: str,
    expected_digest: str,
    snapshot_directory: Path,
    suffix: str,
    max_file_bytes: int,
) -> Path | None:
    source_descriptor = open_regular_cache_entry(
        cache_descriptor,
        entry_name,
        max_file_bytes,
    )
    if source_descriptor is None:
        return None

    snapshot_path = snapshot_directory / f"dataset{suffix}"
    snapshot_descriptor: int | None = None
    snapshot_valid = False
    try:
        snapshot_descriptor = os.open(
            snapshot_path,
            nofollow_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        actual_digest = copy_file_descriptor(
            source_descriptor,
            snapshot_descriptor,
            max_copy_bytes=max_file_bytes,
        )
        os.fsync(snapshot_descriptor)
        if actual_digest != expected_digest:
            return None
        os.fchmod(snapshot_descriptor, 0o400)
        snapshot_valid = True
        return snapshot_path
    except OSError:
        raise RuntimeError(
            "remote dataset verified snapshot could not be created safely"
        ) from None
    finally:
        os.close(source_descriptor)
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)
        if not snapshot_valid:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass


def set_remote_dataset_response_timeout(
    response: Any,
    timeout_seconds: float,
) -> bool:
    """Set the next blocking read timeout on urllib or injected responses."""
    candidates = [response]
    response_fp = getattr(response, "fp", None)
    if response_fp is not None:
        candidates.append(response_fp)
        response_raw = getattr(response_fp, "raw", None)
        if response_raw is not None:
            candidates.append(response_raw)
            response_socket = getattr(response_raw, "_sock", None)
            if response_socket is not None:
                candidates.append(response_socket)

    seen_candidates = set()
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id in seen_candidates:
            continue
        seen_candidates.add(candidate_id)
        set_timeout = getattr(candidate, "settimeout", None)
        if callable(set_timeout):
            set_timeout(timeout_seconds)
            return True
    return False


def close_remote_dataset_response(response: Any) -> None:
    close_response = getattr(response, "close", None)
    if callable(close_response):
        try:
            close_response()
        except Exception:
            pass


def open_remote_dataset_response(
    request_url: str,
    *,
    open_url: Callable[..., Any],
    deadline: float,
    monotonic: Callable[[], float],
) -> Any:
    """Open a URL without allowing DNS, redirects, or headers past deadline."""
    remaining_seconds = deadline - monotonic()
    if remaining_seconds <= 0:
        raise TimeoutError("remote dataset open deadline exceeded")
    request_timeout = min(
        REMOTE_DATASET_REQUEST_TIMEOUT_SECONDS,
        remaining_seconds,
    )
    request_deadline = min(deadline, monotonic() + request_timeout)

    result_lock = threading.Lock()
    result_ready = threading.Event()
    cancelled = False
    response_result: Any = None
    error_result: BaseException | None = None

    def open_worker() -> None:
        nonlocal response_result, error_result
        try:
            response = open_url(request_url, timeout=request_timeout)
        except BaseException as error:
            with result_lock:
                if cancelled:
                    return
                error_result = error
                result_ready.set()
            return

        close_late_response = False
        with result_lock:
            if cancelled:
                close_late_response = True
            else:
                response_result = response
                result_ready.set()
        if close_late_response:
            close_remote_dataset_response(response)

    opener_thread = threading.Thread(
        target=open_worker,
        name="aikit-remote-dataset-open",
        daemon=True,
    )
    opener_thread.start()

    remaining_seconds = request_deadline - monotonic()
    opened_in_time = remaining_seconds > 0 and result_ready.wait(
        timeout=remaining_seconds
    )
    timed_out = not opened_in_time or monotonic() >= request_deadline

    response_to_close = None
    with result_lock:
        if timed_out:
            cancelled = True
            response_to_close = response_result
            response_result = None
        else:
            response = response_result
            error = error_result

    if response_to_close is not None:
        close_remote_dataset_response_in_background(response_to_close)
    if timed_out:
        raise TimeoutError("remote dataset open deadline exceeded")
    if error is not None:
        raise error
    if response is None:
        raise RuntimeError("remote dataset opener returned no response")
    return response


def read_remote_dataset_chunk(response: Any) -> bytes:
    """Use one underlying urllib read, with a generic injected fallback."""
    read_one = getattr(response, "read1", None)
    if callable(read_one):
        return read_one(REMOTE_DATASET_CHUNK_SIZE)
    return response.read(REMOTE_DATASET_CHUNK_SIZE)


class RemoteDatasetReadDeadlineExceeded(TimeoutError):
    pass


def close_remote_dataset_response_in_background(response: Any) -> None:
    threading.Thread(
        target=close_remote_dataset_response,
        args=(response,),
        name="aikit-remote-dataset-close",
        daemon=True,
    ).start()


class RemoteDatasetChunkReader:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.request_ready = threading.Event()
        self.result_ready = threading.Event()
        self.cancelled = threading.Event()
        self.result_lock = threading.Lock()
        self.chunk_result: Any = None
        self.error_result: BaseException | None = None
        self.reader_thread = threading.Thread(
            target=self.read_worker,
            name="aikit-remote-dataset-read",
            daemon=True,
        )
        self.reader_thread.start()

    def cancel(self) -> None:
        self.cancelled.set()
        self.request_ready.set()

    def read_worker(self) -> None:
        while True:
            self.request_ready.wait()
            self.request_ready.clear()
            if self.cancelled.is_set():
                return

            try:
                chunk = read_remote_dataset_chunk(self.response)
            except BaseException as error:
                with self.result_lock:
                    if self.cancelled.is_set():
                        return
                    self.error_result = error
                    self.result_ready.set()
                return

            with self.result_lock:
                if self.cancelled.is_set():
                    return
                self.chunk_result = chunk
                self.result_ready.set()
            if not chunk:
                return

    def read_before_deadline(
        self,
        *,
        deadline: float,
        monotonic: Callable[[], float],
    ) -> bytes:
        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            self.cancel()
            close_remote_dataset_response_in_background(self.response)
            raise RemoteDatasetReadDeadlineExceeded(
                "remote dataset read deadline exceeded"
            )

        with self.result_lock:
            self.chunk_result = None
            self.error_result = None
            self.result_ready.clear()
        self.request_ready.set()

        remaining_seconds = deadline - monotonic()
        read_in_time = remaining_seconds > 0 and self.result_ready.wait(
            timeout=remaining_seconds
        )
        timed_out = not read_in_time or monotonic() >= deadline
        if timed_out:
            self.cancel()
            close_remote_dataset_response_in_background(self.response)
            raise RemoteDatasetReadDeadlineExceeded(
                "remote dataset read deadline exceeded"
            )

        with self.result_lock:
            chunk = self.chunk_result
            error = self.error_result
        if error is not None:
            raise error
        return chunk


def validate_remote_dataset_download_limits(
    total_timeout_seconds: float,
    max_download_bytes: int,
) -> None:
    if (
        isinstance(total_timeout_seconds, bool)
        or not isinstance(total_timeout_seconds, (int, float))
        or not math.isfinite(total_timeout_seconds)
        or total_timeout_seconds <= 0
    ):
        raise ValueError("remote dataset total download timeout must be positive")
    if (
        isinstance(max_download_bytes, bool)
        or not isinstance(max_download_bytes, int)
        or max_download_bytes <= 0
    ):
        raise ValueError("remote dataset maximum download size must be positive")


@contextmanager
def download_remote_dataset_file(
    request_url: str,
    *,
    loader_type: str,
    suffix: str,
    open_url: Callable[..., Any],
    monotonic: Callable[[], float] = time.monotonic,
    total_timeout_seconds: float = REMOTE_DATASET_TOTAL_TIMEOUT_SECONDS,
    max_download_bytes: int = REMOTE_DATASET_MAX_DOWNLOAD_BYTES,
) -> Iterator[tuple[int, str]]:
    validate_remote_dataset_download_limits(
        total_timeout_seconds,
        max_download_bytes,
    )

    deadline = monotonic() + total_timeout_seconds
    with tempfile.TemporaryDirectory(
        prefix="aikit-dataset-download-"
    ) as temporary_directory:
        download_path = Path(temporary_directory) / f"dataset{suffix}"
        download_descriptor = os.open(
            download_path,
            nofollow_open_flags(os.O_RDWR | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        digest = hashlib.sha256()
        downloaded_bytes = 0
        try:
            try:
                response = open_remote_dataset_response(
                    request_url,
                    open_url=open_url,
                    deadline=deadline,
                    monotonic=monotonic,
                )
                response_close_deferred = False
                chunk_reader: RemoteDatasetChunkReader | None = None
                try:
                    chunk_reader = RemoteDatasetChunkReader(response)
                    while True:
                        remaining_seconds = deadline - monotonic()
                        if remaining_seconds <= 0:
                            raise TimeoutError(
                                "remote dataset total download deadline exceeded"
                            )
                        read_timeout = min(
                            REMOTE_DATASET_REQUEST_TIMEOUT_SECONDS,
                            remaining_seconds,
                        )
                        set_remote_dataset_response_timeout(
                            response,
                            read_timeout,
                        )
                        read_deadline = min(
                            deadline,
                            monotonic() + read_timeout,
                        )
                        try:
                            chunk = chunk_reader.read_before_deadline(
                                deadline=read_deadline,
                                monotonic=monotonic,
                            )
                        except RemoteDatasetReadDeadlineExceeded:
                            response_close_deferred = True
                            raise
                        if monotonic() >= deadline:
                            raise TimeoutError(
                                "remote dataset total download deadline exceeded"
                            )
                        if not chunk:
                            break
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_download_bytes:
                            raise RuntimeError(
                                "remote dataset maximum download size exceeded"
                            )
                        write_file_descriptor(download_descriptor, chunk)
                        digest.update(chunk)
                finally:
                    if chunk_reader is not None:
                        chunk_reader.cancel()
                    if not response_close_deferred:
                        close_remote_dataset_response(response)
                os.fsync(download_descriptor)
            except Exception:
                raise RuntimeError(
                    f"remote {loader_type} dataset could not be downloaded"
                ) from None
            yield download_descriptor, digest.hexdigest()
        finally:
            os.close(download_descriptor)


def publish_cached_dataset_file(
    cache_descriptor: int,
    entry_name: str,
    source_descriptor: int,
    expected_digest: str,
    max_file_bytes: int,
) -> None:
    temporary_name = (
        f".publish-{expected_digest}-{secrets.token_hex(16)}.tmp"
    )
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = os.open(
            temporary_name,
            nofollow_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
            dir_fd=cache_descriptor,
        )
        actual_digest = copy_file_descriptor(
            source_descriptor,
            temporary_descriptor,
            max_copy_bytes=max_file_bytes,
        )
        os.fsync(temporary_descriptor)
        if actual_digest != expected_digest:
            raise RuntimeError(
                "remote dataset cache publication verification failed"
            ) from None
        os.replace(
            temporary_name,
            entry_name,
            src_dir_fd=cache_descriptor,
            dst_dir_fd=cache_descriptor,
        )
    except OSError:
        raise RuntimeError(
            "remote dataset cache entry could not be published safely"
        ) from None
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=cache_descriptor)
        except FileNotFoundError:
            pass


def materialize_locked_dataset_snapshot(
    cache_descriptor: int,
    entry_name: str,
    expected_digest: str,
    snapshot_directory: Path,
    suffix: str,
    max_file_bytes: int,
    downloaded_descriptor: int | None = None,
) -> Path | None:
    snapshot_path = create_verified_dataset_snapshot(
        cache_descriptor,
        entry_name,
        expected_digest,
        snapshot_directory,
        suffix,
        max_file_bytes,
    )
    if snapshot_path is not None or downloaded_descriptor is None:
        return snapshot_path

    publish_cached_dataset_file(
        cache_descriptor,
        entry_name,
        downloaded_descriptor,
        expected_digest,
        max_file_bytes,
    )
    snapshot_path = create_verified_dataset_snapshot(
        cache_descriptor,
        entry_name,
        expected_digest,
        snapshot_directory,
        suffix,
        max_file_bytes,
    )
    if snapshot_path is None:
        raise RuntimeError(
            "remote dataset cache verification failed"
        ) from None
    return snapshot_path


@contextmanager
def materialize_remote_dataset_file(
    source: str,
    *,
    loader_type: str,
    checksum: str | None,
    cache_directory: Path | str | None = None,
    open_url: Callable[..., Any] = urlopen,
    monotonic: Callable[[], float] = time.monotonic,
    total_timeout_seconds: float = REMOTE_DATASET_TOTAL_TIMEOUT_SECONDS,
    max_download_bytes: int = REMOTE_DATASET_MAX_DOWNLOAD_BYTES,
) -> Iterator[Path]:
    if loader_type not in REMOTE_DATASET_LOADERS:
        raise ValueError(f"unsupported remote dataset loader {loader_type!r}")
    if not is_http_dataset_source(source):
        raise ValueError(
            f"{loader_type} dataset loader requires an absolute HTTP(S) source"
        )
    if checksum is not None and not DATASET_CHECKSUM_PATTERN.fullmatch(checksum):
        raise ValueError(
            "remote-file dataset loader checksum must use lowercase "
            "sha256:<64 hex> format"
        )
    validate_remote_dataset_download_limits(
        total_timeout_seconds,
        max_download_bytes,
    )

    cache_path = (
        Path(cache_directory)
        if cache_directory is not None
        else dataset_cache_directory()
    )
    suffix = remote_dataset_file_suffix(source, loader_type)
    expected_digest = checksum.removeprefix("sha256:") if checksum else None
    parsed_source = urlparse(source)
    request_url = urlunparse(parsed_source._replace(fragment=""))

    with tempfile.TemporaryDirectory(
        prefix="aikit-verified-dataset-"
    ) as snapshot_directory_name:
        snapshot_directory = Path(snapshot_directory_name)
        os.chmod(snapshot_directory, 0o700)
        with open_dataset_cache_directory(cache_path) as cache_descriptor:
            if expected_digest is not None:
                entry_name = f"{expected_digest}{suffix}"
                with dataset_digest_lock(cache_descriptor, expected_digest):
                    snapshot_path = materialize_locked_dataset_snapshot(
                        cache_descriptor,
                        entry_name,
                        expected_digest,
                        snapshot_directory,
                        suffix,
                        max_download_bytes,
                    )
                if snapshot_path is None:
                    with download_remote_dataset_file(
                        request_url,
                        loader_type=loader_type,
                        suffix=suffix,
                        open_url=open_url,
                        monotonic=monotonic,
                        total_timeout_seconds=total_timeout_seconds,
                        max_download_bytes=max_download_bytes,
                    ) as (download_descriptor, actual_digest):
                        if actual_digest != expected_digest:
                            raise RuntimeError(
                                f"remote {loader_type} dataset checksum "
                                "does not match the configured sha256 digest"
                            ) from None
                        with dataset_digest_lock(
                            cache_descriptor,
                            expected_digest,
                        ):
                            snapshot_path = materialize_locked_dataset_snapshot(
                                cache_descriptor,
                                entry_name,
                                expected_digest,
                                snapshot_directory,
                                suffix,
                                max_download_bytes,
                                downloaded_descriptor=download_descriptor,
                            )
            else:
                with download_remote_dataset_file(
                    request_url,
                    loader_type=loader_type,
                    suffix=suffix,
                    open_url=open_url,
                    monotonic=monotonic,
                    total_timeout_seconds=total_timeout_seconds,
                    max_download_bytes=max_download_bytes,
                ) as (download_descriptor, actual_digest):
                    entry_name = f"{actual_digest}{suffix}"
                    with dataset_digest_lock(cache_descriptor, actual_digest):
                        snapshot_path = materialize_locked_dataset_snapshot(
                            cache_descriptor,
                            entry_name,
                            actual_digest,
                            snapshot_directory,
                            suffix,
                            max_download_bytes,
                            downloaded_descriptor=download_descriptor,
                        )

        yield snapshot_path


def load_training_dataset(
    dataset_spec: TrainingDatasetSpec,
    *,
    load_dataset: Callable[..., Any],
    cache_directory: Path | str | None = None,
) -> Any:
    loader_type = dataset_spec.loader.loader_type
    is_remote = is_http_dataset_source(dataset_spec.source) and (
        loader_type is None or loader_type in REMOTE_DATASET_LOADERS
    )

    def load_materialized_dataset(local_file: Path | None = None) -> Any:
        load_spec = dataset_load_spec(dataset_spec, local_file=local_file)
        try:
            return load_dataset(load_spec.path, **load_spec.kwargs)
        except Exception:
            subject = dataset_error_subject(
                dataset_spec.dataset_type,
                source=dataset_spec.source,
                loader_type=loader_type,
                dataset_index=dataset_spec.index,
            )
            raise RuntimeError(f"{subject} could not be loaded") from None

    try:
        if not is_remote:
            return load_materialized_dataset()

        effective_loader = loader_type or DATASET_LOADER_JSON
        with materialize_remote_dataset_file(
            dataset_spec.source,
            loader_type=effective_loader,
            checksum=dataset_spec.loader.checksum,
            cache_directory=cache_directory,
        ) as local_file:
            return load_materialized_dataset(local_file)
    except (OSError, RuntimeError, ValueError) as error:
        if dataset_spec.index is None or str(error).startswith(
            f"datasets[{dataset_spec.index}] "
        ):
            raise
        raise type(error)(
            f"datasets[{dataset_spec.index}] {error}"
        ) from None

def configured_training_loss(train_config: Mapping[str, Any]) -> str:
    cfg = unsloth_config(train_config)
    configured_loss = cfg.get("loss", LOSS_ALL)
    loss = LOSS_ALL if configured_loss is None else configured_loss
    if not isinstance(loss, str) or loss not in SUPPORTED_LOSSES:
        raise ValueError(f"unsupported SFT loss {loss!r}")
    return loss


def dataset_compatibility_group(dataset_type: str, *, loss: str) -> str:
    if dataset_type == DATASET_TYPE_PREFERENCE:
        raise ValueError(
            "preference datasets are supported only for the DPO objective"
        )
    if loss == LOSS_RESPONSE and dataset_type not in CHAT_DATASET_TYPES:
        raise ValueError(
            "response SFT loss is supported only for messages and sharegpt datasets"
        )
    if loss == LOSS_RESPONSE:
        return DATASET_COMPATIBILITY_RESPONSE_CHAT
    if dataset_type == DATASET_TYPE_PROMPT_COMPLETION:
        return DATASET_COMPATIBILITY_PROMPT_COMPLETION
    return DATASET_COMPATIBILITY_FULL_SEQUENCE


def training_dataset_compatibility(
    dataset_specs: Sequence[TrainingDatasetSpec],
    *,
    loss: str,
) -> str:
    if not dataset_specs:
        raise ValueError("training configuration must define at least one dataset")

    first_spec = dataset_specs[0]
    try:
        first_group = dataset_compatibility_group(
            first_spec.dataset_type,
            loss=loss,
        )
    except ValueError as error:
        raise ValueError(
            f"datasets[0] type {first_spec.dataset_type}: {error}"
        ) from None

    for dataset_index, dataset_spec in enumerate(dataset_specs[1:], start=1):
        try:
            group = dataset_compatibility_group(
                dataset_spec.dataset_type,
                loss=loss,
            )
        except ValueError as error:
            raise ValueError(
                f"datasets[{dataset_index}] type {dataset_spec.dataset_type}: {error}"
            ) from None
        if group != first_group:
            raise ValueError(
                f"datasets[{dataset_index}] type {dataset_spec.dataset_type} is "
                f"incompatible with datasets[0] type {first_spec.dataset_type}: "
                f"{group} and {first_group} datasets cannot be combined"
            )
    return first_group


def validate_response_packing(
    train_config: Mapping[str, Any],
    *,
    loss: str,
) -> None:
    cfg = unsloth_config(train_config)
    if loss == LOSS_RESPONSE and bool(cfg.get("packing", False)):
        raise ValueError(
            "response SFT loss does not support packing because response masks "
            "must not cross conversation boundaries; set config.unsloth.packing "
            "to false"
        )


def training_loss(
    train_config: Mapping[str, Any],
    *,
    dataset_type: str,
) -> str:
    loss = configured_training_loss(train_config)
    dataset_compatibility_group(dataset_type, loss=loss)
    validate_response_packing(train_config, loss=loss)
    return loss


def validate_training_objective(
    train_config: Mapping[str, Any],
    *,
    objective: TrainingObjectiveSpec,
    dataset_spec: TrainingDatasetSpec,
) -> str:
    if objective.objective_type == OBJECTIVE_TYPE_SFT:
        if dataset_spec.dataset_type == DATASET_TYPE_PREFERENCE:
            raise ValueError(
                "preference datasets are supported only for the DPO objective"
            )
        return training_loss(
            train_config,
            dataset_type=dataset_spec.dataset_type,
        )

    if dataset_spec.dataset_type != DATASET_TYPE_PREFERENCE:
        raise ValueError(
            "DPO objective requires a preference dataset with prompt, chosen, "
            "and rejected fields"
        )
    if (
        dataset_spec.loader.loader_type == DATASET_LOADER_TEXT
    ):
        raise ValueError(
            "preference datasets do not support the text loader because DPO "
            "requires prompt, chosen, and rejected columns"
        )

    cfg = unsloth_config(train_config)
    packing = cfg.get("packing", False)
    if not isinstance(packing, bool):
        raise ValueError("config.unsloth.packing must be a boolean")
    if packing:
        raise ValueError("DPO objective does not support packing")

    configured_loss = cfg.get("loss", LOSS_ALL)
    loss = LOSS_ALL if configured_loss is None else configured_loss
    if not isinstance(loss, str) or loss not in SUPPORTED_LOSSES:
        raise ValueError(f"unsupported SFT loss {loss!r}")
    if loss == LOSS_RESPONSE:
        raise ValueError(
            "response SFT loss is not supported for the DPO objective"
        )

    max_seq_length = cfg.get("maxSeqLength")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length <= 0
    ):
        raise ValueError(
            "config.unsloth.maxSeqLength must be an integer greater than zero"
        )
    if (
        objective.max_prompt_length is None
        or objective.max_prompt_length > max_seq_length
    ):
        raise ValueError(
            "DPO objective maxPromptLength must not exceed "
            "config.unsloth.maxSeqLength"
        )
    return LOSS_ALL


def dataset_error_subject(
    dataset_type: str,
    *,
    source: str | None = None,
    record_index: int | None = None,
    loader_type: str | None = None,
    dataset_index: int | None = None,
) -> str:
    if source is None:
        subject = f"{dataset_type} dataset"
        if record_index is not None:
            subject += f" record {record_index}"
    else:
        subject = (
            f"{dataset_type} dataset "
            f"{dataset_source_description(source, loader_type=loader_type)}"
        )
        if record_index is not None:
            subject += f" row {record_index}"
    if dataset_index is not None:
        subject = f"datasets[{dataset_index}] {subject}"
    return subject


@contextmanager
def indexed_dataset_errors(dataset_index: int) -> Iterator[None]:
    try:
        yield
    except (RuntimeError, ValueError) as error:
        prefix = f"datasets[{dataset_index}] "
        if str(error).startswith(prefix):
            raise
        raise type(error)(f"{prefix}{error}") from None


def validate_messages_value(
    messages: Any,
    *,
    subject: str,
) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f'{subject} field "messages" must be a non-empty list')

    has_assistant = False
    for message_index, message in enumerate(messages):
        message_subject = f"{subject} message {message_index}"
        if not isinstance(message, Mapping):
            raise ValueError(f"{message_subject} must be a mapping")

        missing_fields = [
            field for field in MESSAGE_FIELDS if field not in message
        ]
        if missing_fields:
            quoted_fields = ", ".join(
                f'"{field}"' for field in sorted(missing_fields)
            )
            raise ValueError(
                f"{message_subject} is missing required fields: {quoted_fields}"
            )

        unsupported_fields = [
            field for field in message if field not in MESSAGE_FIELDS
        ]
        if unsupported_fields:
            quoted_fields = ", ".join(
                sorted(repr(field) for field in unsupported_fields)
            )
            raise ValueError(
                f"{message_subject} contains unsupported fields: {quoted_fields}"
            )

        role = message["role"]
        content = message["content"]
        if not isinstance(role, str):
            raise ValueError(f'{message_subject} field "role" must be a string')
        if role not in SUPPORTED_MESSAGE_ROLES:
            raise ValueError(
                f"{message_subject} has unsupported role {role!r}; "
                "supported roles are system, user, and assistant"
            )
        if not isinstance(content, str):
            raise ValueError(
                f'{message_subject} field "content" must be a string'
            )
        if role == "assistant":
            has_assistant = True

    if not has_assistant:
        raise ValueError(f"{subject} must contain at least one assistant message")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{subject} final message must have role 'assistant'")


def canonicalize_sharegpt_conversations(
    conversations: Any,
    *,
    subject: str,
) -> list[dict[str, str]]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(
            f'{subject} field "conversations" must be a non-empty list'
        )

    messages = []
    for message_index, conversation in enumerate(conversations):
        message_subject = f"{subject} conversation {message_index}"
        if not isinstance(conversation, Mapping):
            raise ValueError(f"{message_subject} must be a mapping")

        missing_fields = [
            field for field in SHAREGPT_MESSAGE_FIELDS if field not in conversation
        ]
        if missing_fields:
            quoted_fields = ", ".join(
                f'"{field}"' for field in sorted(missing_fields)
            )
            raise ValueError(
                f"{message_subject} is missing required fields: {quoted_fields}"
            )

        unsupported_fields = [
            field for field in conversation if field not in SHAREGPT_MESSAGE_FIELDS
        ]
        if unsupported_fields:
            quoted_fields = ", ".join(
                sorted(repr(field) for field in unsupported_fields)
            )
            raise ValueError(
                f"{message_subject} contains unsupported fields: {quoted_fields}"
            )

        source_role = conversation["from"]
        content = conversation["value"]
        if not isinstance(source_role, str):
            raise ValueError(
                f'{message_subject} field "from" must be a string'
            )
        try:
            role = SHAREGPT_ROLE_MAP[source_role]
        except KeyError:
            raise ValueError(
                f"{message_subject} has unsupported role {source_role!r}; "
                "supported roles are system, human, user, gpt, and assistant"
            ) from None
        if not isinstance(content, str):
            raise ValueError(
                f'{message_subject} field "value" must be a string'
            )
        messages.append({"role": role, "content": content})

    validate_messages_value(messages, subject=subject)
    return messages


def validate_messages_top_level_fields(
    fields: Sequence[Any] | Mapping[Any, Any],
    *,
    subject: str,
) -> None:
    unsupported_fields = [
        field
        for field in fields
        if field in UNSUPPORTED_MESSAGES_TOP_LEVEL_FIELDS
    ]
    if unsupported_fields:
        quoted_fields = ", ".join(
            sorted(repr(field) for field in unsupported_fields)
        )
        raise ValueError(
            f"{subject} contains unsupported top-level fields: {quoted_fields}"
        )


def validate_training_dataset(
    dataset: Any,
    *,
    dataset_type: str,
    source: str | None = None,
) -> None:
    required_fields = DATASET_REQUIRED_FIELDS[dataset_type]
    record_count = 0

    for record_index, record in enumerate(dataset):
        record_count += 1
        subject = dataset_error_subject(
            dataset_type,
            source=source,
            record_index=record_index,
        )
        if not isinstance(record, Mapping):
            raise ValueError(f"{subject} must be a mapping")

        if dataset_type in CHAT_DATASET_TYPES:
            validate_messages_top_level_fields(record, subject=subject)

        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f'{subject} is missing required field "{field}"'
                )

            value = record[field]
            if dataset_type in CHAT_DATASET_TYPES:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f'{subject} field "{field}" must be a string'
                )

        if dataset_type == DATASET_TYPE_MESSAGES:
            validate_messages_value(record["messages"], subject=subject)
        if dataset_type == DATASET_TYPE_SHAREGPT:
            canonicalize_sharegpt_conversations(
                record["conversations"],
                subject=subject,
            )
        if (
            dataset_type == DATASET_TYPE_PROMPT_COMPLETION
            and record["completion"] == ""
        ):
            raise ValueError(
                f'{dataset_type} dataset record {record_index} field "completion" must be a non-empty string'
            )
        if dataset_type == DATASET_TYPE_PREFERENCE:
            for field in DATASET_REQUIRED_FIELDS[DATASET_TYPE_PREFERENCE]:
                if not record[field].strip():
                    raise ValueError(
                        f'{subject} field "{field}" must be a non-empty string'
                    )
            if record["chosen"] == record["rejected"]:
                raise ValueError(
                    f'{subject} fields "chosen" and "rejected" must be distinct'
                )
        if dataset_type == DATASET_TYPE_TEXT and record["text"] == "":
            raise ValueError(
                f'{subject} field "text" must be a non-empty string'
            )

    if record_count == 0:
        subject = dataset_error_subject(dataset_type, source=source)
        raise ValueError(f"{subject} must contain at least one record")


def project_training_dataset(
    dataset: Any,
    *,
    dataset_type: str,
    source: str | None = None,
) -> Any:
    subject = dataset_error_subject(dataset_type, source=source)
    if len(dataset) == 0:
        raise ValueError(f"{subject} must contain at least one record")

    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, Sequence) or isinstance(
        column_names, (str, bytes)
    ):
        raise ValueError(f"{subject} does not expose its columns")

    required_fields = DATASET_REQUIRED_FIELDS[dataset_type]
    missing_fields = [
        field for field in required_fields if field not in column_names
    ]
    if missing_fields:
        quoted_fields = ", ".join(f'"{field}"' for field in missing_fields)
        raise ValueError(
            f"{subject} is missing required columns: {quoted_fields}"
        )

    if dataset_type in CHAT_DATASET_TYPES:
        validate_messages_top_level_fields(column_names, subject=subject)

    return dataset.select_columns(list(required_fields))


def normalize_sharegpt_example(
    example: Mapping[str, Any],
    raw_index: Any,
    *,
    source_description: str,
) -> dict[str, list[dict[str, str]]]:
    try:
        record_index = operator.index(raw_index)
    except TypeError as error:
        raise ShareGPTNormalizationError(
            "sharegpt normalization row index must be an integer"
        ) from error
    subject = (
        f"{DATASET_TYPE_SHAREGPT} dataset {source_description} "
        f"row {record_index}"
    )
    try:
        messages = canonicalize_sharegpt_conversations(
            example["conversations"],
            subject=subject,
        )
    except (KeyError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else (
            f'{subject} is missing required field "conversations"'
        )
        raise ShareGPTNormalizationError(message) from None
    return {"messages": messages}


def normalize_sharegpt_dataset(
    dataset: Any,
    *,
    source: str,
) -> Any:
    source_description = dataset_source_description(source)
    try:
        return dataset.map(
            partial(
                normalize_sharegpt_example,
                source_description=source_description,
            ),
            batched=False,
            with_indices=True,
            remove_columns=list(dataset.column_names),
            writer_batch_size=1,
        )
    except ShareGPTNormalizationError:
        raise
    except Exception:
        subject = dataset_error_subject(DATASET_TYPE_SHAREGPT, source=source)
        raise RuntimeError(f"{subject} could not be normalized") from None


def effective_messages_chat_template(processing_class: Any) -> str | None:
    chat_template = getattr(processing_class, "chat_template", None)
    inner_tokenizer = getattr(processing_class, "tokenizer", processing_class)
    if not chat_template and inner_tokenizer is not processing_class:
        chat_template = getattr(inner_tokenizer, "chat_template", None)

    if isinstance(chat_template, str):
        return chat_template if chat_template.strip() else None
    if isinstance(chat_template, Mapping):
        default_template = chat_template.get("default")
        if isinstance(default_template, str) and default_template.strip():
            return default_template
    return None


def require_messages_chat_template(processing_class: Any) -> None:
    if not callable(getattr(processing_class, "apply_chat_template", None)):
        raise RuntimeError(
            "messages preprocessing requires a tokenizer with apply_chat_template"
        )

    chat_template = effective_messages_chat_template(processing_class)
    if chat_template is None:
        raise RuntimeError(
            "messages preprocessing requires a usable tokenizer chat template; "
            "use an instruct/chat model that defines tokenizer.chat_template"
        )
    if "strftime_now" in chat_template:
        raise RuntimeError(
            "messages preprocessing does not support wall-clock-dependent "
            "tokenizer chat templates containing strftime_now until "
            "deterministic template values are configured and included in "
            "cache keys"
        )


def render_messages_example(
    example: Mapping[str, Any],
    raw_index: Any,
    *,
    processing_class: Any,
    source_description: str,
    dataset_type: str = DATASET_TYPE_MESSAGES,
) -> dict[str, str]:
    try:
        record_index = operator.index(raw_index)
    except TypeError as error:
        raise MessagesRenderError(
            "messages preprocessing row index must be an integer"
        ) from error
    subject = (
        f"{dataset_type} dataset {source_description} "
        f"row {record_index}"
    )
    try:
        text = processing_class.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Do not propagate template errors that may echo sensitive source data.
        raise MessagesRenderError(
            f"{subject} could not be rendered with the tokenizer chat template"
        ) from None
    if not isinstance(text, str) or not text:
        raise MessagesRenderError(
            f"{subject} tokenizer chat template must render a non-empty string"
        )
    return {"text": text}


def render_messages_dataset(
    dataset: Any,
    *,
    processing_class: Any,
    source: str,
    dataset_type: str = DATASET_TYPE_MESSAGES,
) -> Any:
    source_description = dataset_source_description(source)
    try:
        return dataset.map(
            partial(
                render_messages_example,
                processing_class=processing_class,
                source_description=source_description,
                dataset_type=dataset_type,
            ),
            batched=False,
            with_indices=True,
            remove_columns=list(dataset.column_names),
            writer_batch_size=1,
        )
    except MessagesRenderError:
        raise
    except Exception:
        subject = dataset_error_subject(dataset_type, source=source)
        raise RuntimeError(
            f"{subject} could not be rendered with the tokenizer chat template"
        ) from None


def messages_unsloth_add_special_tokens(
    processing_class: Any,
    first_text: str,
) -> bool:
    """Match the locked Unsloth rendered-text special-token decision."""
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    chat_template = effective_messages_chat_template(processing_class) or ""
    bos_token = getattr(processing_class, "bos_token", None) or getattr(
        tokenizer,
        "bos_token",
        None,
    )
    return not (
        bos_token is not None
        and (first_text.startswith(bos_token) or bos_token in chat_template)
    )


def messages_chat_template_token_ids(
    processing_class: Any,
    messages: list[Mapping[str, str]],
    *,
    max_length: int,
    subject: str,
) -> list[Any]:
    try:
        tokenized = processing_class.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
            return_dict=False,
        )
    except Exception:
        raise RuntimeError(
            f"{subject} could not be tokenized with the tokenizer chat template"
        ) from None

    if isinstance(tokenized, Mapping):
        try:
            tokenized = tokenized["input_ids"]
        except KeyError as error:
            raise RuntimeError(
                f"{subject} chat-template tokenization did not produce input_ids"
            ) from error
    input_ids = sequence_values(
        tokenized,
        description="chat-template token IDs",
        preprocessing="messages",
    )
    if input_ids and isinstance(input_ids[0], Sequence):
        raise RuntimeError(
            f"{subject} chat-template token IDs must be one-dimensional"
        )
    if len(input_ids) > max_length:
        raise RuntimeError(
            f"{subject} chat-template tokenization did not honor its truncation limit"
        )
    return input_ids


def messages_rendered_token_id_rows(
    processing_class: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
    max_length: int,
    subject: str,
) -> list[list[Any]]:
    try:
        tokenized = processing_class(
            list(texts),
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    except Exception:
        raise RuntimeError(f"{subject} could not be tokenized") from None
    if not isinstance(tokenized, Mapping):
        raise RuntimeError(f"{subject} tokenization must return a mapping")
    try:
        token_rows = sequence_values(
            tokenized["input_ids"],
            description=f"{subject} token ID rows",
            preprocessing="messages",
        )
    except KeyError as error:
        raise RuntimeError(
            f"{subject} tokenization did not produce input_ids"
        ) from error
    if len(token_rows) != len(texts):
        raise RuntimeError(
            f"{subject} tokenization returned an unexpected number of rows"
        )

    input_id_rows = []
    for row_index, input_ids in enumerate(token_rows):
        row = sequence_values(
            input_ids,
            description=f"{subject} token IDs row {row_index}",
            preprocessing="messages",
        )
        if len(row) > max_length:
            raise RuntimeError(
                f"{subject} tokenization did not honor its truncation limit"
            )
        input_id_rows.append(row)
    return input_id_rows


def empty_messages_token_fingerprint() -> MessagesTokenFingerprint:
    return MessagesTokenFingerprint(0, 0, 0)


def merge_messages_token_fingerprints(
    fingerprints: Sequence[MessagesTokenFingerprint],
) -> MessagesTokenFingerprint:
    merged = empty_messages_token_fingerprint()
    for fingerprint in fingerprints:
        merged = MessagesTokenFingerprint(
            merged.sequence_count + fingerprint.sequence_count,
            (merged.first_digest_sum + fingerprint.first_digest_sum)
            & MESSAGES_TOKEN_FINGERPRINT_MASK,
            (merged.second_digest_sum + fingerprint.second_digest_sum)
            & MESSAGES_TOKEN_FINGERPRINT_MASK,
        )
    return merged


def extend_messages_token_fingerprint(
    fingerprint: MessagesTokenFingerprint,
    token_ids: Sequence[Any],
    *,
    description: str,
) -> MessagesTokenFingerprint:
    encoded_token_ids = bytearray()
    for token_index, raw_token_id in enumerate(token_ids):
        try:
            token_id = operator.index(raw_token_id)
        except TypeError as error:
            raise RuntimeError(
                f"messages preprocessing {description} token ID {token_index} must be an integer"
            ) from error
        if isinstance(raw_token_id, bool) or not 0 <= token_id < 1 << 64:
            raise RuntimeError(
                f"messages preprocessing {description} token ID {token_index} must be an unsigned 64-bit integer"
            )
        encoded_token_ids.extend(token_id.to_bytes(8, byteorder="big"))

    first_hasher = hashlib.sha256(b"aikit-messages-token-sequence-1\x00")
    first_hasher.update(encoded_token_ids)
    second_hasher = hashlib.sha256(b"aikit-messages-token-sequence-2\x00")
    second_hasher.update(encoded_token_ids)
    first_digest = int.from_bytes(first_hasher.digest(), byteorder="big")
    second_digest = int.from_bytes(second_hasher.digest(), byteorder="big")

    return MessagesTokenFingerprint(
        fingerprint.sequence_count + 1,
        (fingerprint.first_digest_sum + first_digest)
        & MESSAGES_TOKEN_FINGERPRINT_MASK,
        (fingerprint.second_digest_sum + second_digest)
        & MESSAGES_TOKEN_FINGERPRINT_MASK,
    )


def validate_messages_tokenization(
    source_dataset: Any,
    rendered_dataset: Any,
    *,
    processing_class: Any,
    max_seq_length: int,
    source: str,
    batch_size: int = MESSAGES_VALIDATION_BATCH_SIZE,
    dataset_type: str = DATASET_TYPE_MESSAGES,
    response_markers: ResponseMarkers | None = None,
) -> MessagesTokenFingerprint:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("messages validation batch size must be positive")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length <= 0
    ):
        raise ValueError("messages max sequence length must be positive")

    try:
        first_rendered_record = next(iter(rendered_dataset))
    except StopIteration as error:
        raise RuntimeError(
            "messages preprocessing produced an empty rendered dataset"
        ) from error
    if not isinstance(first_rendered_record, Mapping) or not isinstance(
        first_rendered_record.get("text"), str
    ):
        raise RuntimeError(
            "messages preprocessing rendered records must contain string text"
        )
    add_special_tokens = messages_unsloth_add_special_tokens(
        processing_class,
        first_rendered_record["text"],
    )
    unsloth_tokenizer = getattr(
        processing_class,
        "tokenizer",
        processing_class,
    )

    token_limit = max_seq_length + 1
    estimated_tokens_per_record = token_limit * 2
    effective_batch_size = min(
        batch_size,
        max(
            1,
            MESSAGES_VALIDATION_TOKEN_BUDGET
            // estimated_tokens_per_record,
        ),
    )
    fingerprint = empty_messages_token_fingerprint()
    record_count = 0
    batch_start = 0
    rendered_texts: list[str] = []
    canonical_message_rows: list[Sequence[Mapping[str, str]]] = []
    canonical_token_rows: list[list[Any]] = []

    def validate_batch() -> None:
        nonlocal fingerprint
        batch_subject = dataset_error_subject(
            dataset_type,
            source=source,
        )
        batch_subject += (
            f" rows {batch_start}-{batch_start + len(rendered_texts) - 1}"
        )
        rendered_token_rows = messages_rendered_token_id_rows(
            unsloth_tokenizer,
            rendered_texts,
            add_special_tokens=add_special_tokens,
            max_length=token_limit,
            subject=batch_subject,
        )
        for batch_index, (messages, canonical_ids, rendered_ids) in enumerate(
            zip(
                canonical_message_rows,
                canonical_token_rows,
                rendered_token_rows,
            )
        ):
            record_index = batch_start + batch_index
            subject = dataset_error_subject(
                dataset_type,
                source=source,
                record_index=record_index,
            )
            if len(canonical_ids) > max_seq_length:
                raise ValueError(
                    f"{subject} produces at least {len(canonical_ids)} tokens "
                    f"after chat-template rendering, exceeding maxSeqLength "
                    f"{max_seq_length}; training truncation would discard part "
                    "of the conversation"
                )
            if not canonical_ids:
                raise ValueError(
                    f"{subject} chat template must produce at least one token"
                )
            if canonical_ids != rendered_ids:
                raise RuntimeError(
                    f"{subject} canonical chat-template token IDs do not match "
                    "the locked Unsloth rendered-text tokenization"
                )
            if response_markers is not None:
                validate_response_marker_layout(
                    messages,
                    canonical_ids,
                    markers=response_markers,
                    subject=subject,
                )
            fingerprint = extend_messages_token_fingerprint(
                fingerprint,
                canonical_ids,
                description=f"source record {record_index}",
            )

    sentinel = object()
    source_iterator = iter(source_dataset)
    rendered_iterator = iter(rendered_dataset)
    while True:
        source_record = next(source_iterator, sentinel)
        rendered_record = next(rendered_iterator, sentinel)
        if source_record is sentinel and rendered_record is sentinel:
            break
        if source_record is sentinel or rendered_record is sentinel:
            raise RuntimeError(
                "messages preprocessing rendered record count does not match the source"
            )
        if not isinstance(source_record, Mapping) or not isinstance(
            rendered_record, Mapping
        ):
            raise RuntimeError(
                "messages preprocessing source and rendered records must be mappings"
            )
        text = rendered_record.get("text")
        if not isinstance(text, str):
            raise RuntimeError(
                "messages preprocessing rendered records must contain string text"
            )
        subject = dataset_error_subject(
            dataset_type,
            source=source,
            record_index=record_count,
        )
        messages = source_record["messages"]
        canonical_ids = messages_chat_template_token_ids(
            processing_class,
            messages,
            max_length=token_limit,
            subject=subject,
        )
        rendered_texts.append(text)
        canonical_message_rows.append(messages)
        canonical_token_rows.append(canonical_ids)
        record_count += 1
        if len(rendered_texts) == effective_batch_size:
            validate_batch()
            rendered_texts.clear()
            canonical_message_rows.clear()
            canonical_token_rows.clear()
            batch_start = record_count

    if rendered_texts:
        validate_batch()
    if record_count == 0:
        raise RuntimeError(
            "messages preprocessing produced an empty source dataset"
        )
    return fingerprint


def text_token_ids(
    processing_class: Any,
    text: str,
    *,
    add_special_tokens: bool,
    description: str,
) -> list[Any]:
    try:
        tokenized = processing_class(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"text preprocessing could not tokenize {description}"
        ) from error

    if not isinstance(tokenized, Mapping):
        raise RuntimeError(
            f"text preprocessing {description} tokenization must return a mapping"
        )

    try:
        input_ids = tokenized["input_ids"]
    except KeyError as error:
        raise RuntimeError(
            f"text preprocessing {description} tokenization did not produce input_ids"
        ) from error

    to_list = getattr(input_ids, "tolist", None)
    if callable(to_list):
        input_ids = to_list()
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise RuntimeError(
            f"text preprocessing {description} token IDs must be a sequence"
        )
    if input_ids and isinstance(input_ids[0], Sequence):
        raise RuntimeError(
            f"text preprocessing {description} token IDs must be one-dimensional"
        )

    return list(input_ids)


def text_token_id_rows(
    processing_class: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
    max_length: int,
    description: str,
) -> list[list[Any]]:
    try:
        tokenized = processing_class(
            list(texts),
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    except Exception as error:
        raise RuntimeError(
            f"text preprocessing could not tokenize {description}"
        ) from error

    if not isinstance(tokenized, Mapping):
        raise RuntimeError(
            f"text preprocessing {description} tokenization must return a mapping"
        )

    try:
        token_rows = tokenized["input_ids"]
    except KeyError as error:
        raise RuntimeError(
            f"text preprocessing {description} tokenization did not produce input_ids"
        ) from error

    to_list = getattr(token_rows, "tolist", None)
    if callable(to_list):
        token_rows = to_list()
    if not isinstance(token_rows, Sequence) or isinstance(
        token_rows, (str, bytes)
    ):
        raise RuntimeError(
            f"text preprocessing {description} token ID rows must be a sequence"
        )
    if len(token_rows) != len(texts):
        raise RuntimeError(
            f"text preprocessing {description} tokenization returned an unexpected number of rows"
        )

    input_id_rows = []
    for row_index, input_ids in enumerate(token_rows):
        to_list = getattr(input_ids, "tolist", None)
        if callable(to_list):
            input_ids = to_list()
        if not isinstance(input_ids, Sequence) or isinstance(
            input_ids, (str, bytes)
        ):
            raise RuntimeError(
                f"text preprocessing {description} token IDs row {row_index} must be a sequence"
            )
        input_ids = list(input_ids)
        if len(input_ids) > max_length:
            raise RuntimeError(
                f"text preprocessing {description} tokenization did not honor its truncation limit"
            )
        input_id_rows.append(input_ids)

    return input_id_rows


def leading_token_count(input_ids: Sequence[Any], token_id: int) -> int:
    count = 0
    for input_id in input_ids:
        if input_id != token_id:
            break
        count += 1
    return count


def trailing_token_count(input_ids: Sequence[Any], token_id: int) -> int:
    count = 0
    for input_id in reversed(input_ids):
        if input_id != token_id:
            break
        count += 1
    return count


def normalize_text_value(text: str, *, policy: TextBoundaryPolicy) -> str:
    normalized = text
    if policy.bos_token is not None:
        while normalized.startswith(policy.bos_token):
            normalized = normalized[len(policy.bos_token) :]

    while normalized.endswith(policy.eos_token):
        normalized = normalized[: -len(policy.eos_token)]

    if policy.bos_token is not None and not policy.add_special_tokens:
        normalized = policy.bos_token + normalized

    if policy.append_eos_token:
        normalized += policy.eos_token

    return normalized


def normalize_text_examples(
    examples: Mapping[str, Sequence[str]],
    *,
    policy: TextBoundaryPolicy,
) -> dict[str, list[str]]:
    return {
        "text": [
            normalize_text_value(text, policy=policy) for text in examples["text"]
        ]
    }


def text_preprocessing_verification_sources(
    policy: TextBoundaryPolicy,
) -> list[str]:
    sources = list(TEXT_PREPROCESSING_VERIFICATION_TEXTS)
    sources[0] += policy.eos_token * 2
    if policy.bos_token is not None:
        sources[0] = policy.bos_token * 2 + sources[0]
    return sources


def text_boundary_policy(processing_class: Any) -> TextBoundaryPolicy:
    eos_token = getattr(processing_class, "eos_token", None)
    eos_token_id = getattr(processing_class, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError(
            "text preprocessing requires a non-empty tokenizer EOS token"
        )
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool):
        raise RuntimeError(
            "text preprocessing requires an integer tokenizer EOS token ID"
        )
    if text_token_ids(
        processing_class,
        eos_token,
        add_special_tokens=False,
        description="EOS token",
    ) != [eos_token_id]:
        raise RuntimeError(
            "text preprocessing tokenizer EOS token does not encode to its EOS token ID"
        )

    bos_token = getattr(processing_class, "bos_token", None)
    bos_token_id = getattr(processing_class, "bos_token_id", None)
    if bos_token is None and bos_token_id is None:
        bos_token = None
        bos_token_id = None
        tokenizer_adds_bos = False
    else:
        if not isinstance(bos_token, str) or not bos_token:
            raise RuntimeError(
                "text preprocessing tokenizer BOS token must be a non-empty string or absent"
            )
        if not isinstance(bos_token_id, int) or isinstance(bos_token_id, bool):
            raise RuntimeError(
                "text preprocessing tokenizer BOS token ID must be an integer or absent"
            )
        if text_token_ids(
            processing_class,
            bos_token,
            add_special_tokens=False,
            description="BOS token",
        ) != [bos_token_id]:
            raise RuntimeError(
                "text preprocessing tokenizer BOS token does not encode to its BOS token ID"
            )

        auto_bos_results = []
        for index, probe_text in enumerate(TEXT_PREPROCESSING_VERIFICATION_TEXTS):
            without_special_tokens = text_token_ids(
                processing_class,
                probe_text,
                add_special_tokens=False,
                description=f"BOS probe {index} without special tokens",
            )
            with_special_tokens = text_token_ids(
                processing_class,
                probe_text,
                add_special_tokens=True,
                description=f"BOS probe {index} with special tokens",
            )
            auto_bos_results.append(
                leading_token_count(with_special_tokens, bos_token_id)
                > leading_token_count(without_special_tokens, bos_token_id)
            )
        if len(set(auto_bos_results)) != 1:
            raise RuntimeError(
                "text preprocessing tokenizer BOS behavior is inconsistent"
            )
        tokenizer_adds_bos = auto_bos_results[0]

    chat_template = getattr(processing_class, "chat_template", "") or ""
    template_suppresses_special_tokens = (
        bos_token is not None
        and isinstance(chat_template, str)
        and bos_token in chat_template
    )
    # Locked Unsloth-Zoo chooses one add_special_tokens value for the entire
    # dataset from its first row and chat template. Use one representation for
    # every row so that global choice cannot add or omit a BOS in mixed data.
    add_special_tokens = bos_token is None or (
        tokenizer_adds_bos and not template_suppresses_special_tokens
    )
    automatic_eos_counts = [
        trailing_token_count(
            text_token_ids(
                processing_class,
                probe_text,
                add_special_tokens=add_special_tokens,
                description=f"EOS behavior probe {index}",
            ),
            eos_token_id,
        )
        for index, probe_text in enumerate(TEXT_PREPROCESSING_VERIFICATION_TEXTS)
    ]
    if len(set(automatic_eos_counts)) != 1:
        raise RuntimeError(
            "text preprocessing tokenizer automatic EOS behavior is inconsistent"
        )
    automatic_eos_count = automatic_eos_counts[0]
    if automatic_eos_count not in {0, 1}:
        raise RuntimeError(
            "text preprocessing tokenizer must add at most one automatic terminal EOS token"
        )

    policy = TextBoundaryPolicy(
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        bos_token=bos_token,
        bos_token_id=bos_token_id,
        add_special_tokens=add_special_tokens,
        append_eos_token=automatic_eos_count == 0,
    )

    for index, probe_text in enumerate(TEXT_PREPROCESSING_VERIFICATION_TEXTS):
        normalized = normalize_text_value(probe_text, policy=policy)
        input_ids = text_token_ids(
            processing_class,
            normalized,
            add_special_tokens=policy.add_special_tokens,
            description=f"normalized boundary probe {index}",
        )
        if trailing_token_count(input_ids, eos_token_id) != 1:
            raise RuntimeError(
                "text preprocessing must produce exactly one terminal tokenizer EOS token"
            )
        if bos_token_id is not None and leading_token_count(
            input_ids, bos_token_id
        ) != 1:
            raise RuntimeError(
                "text preprocessing must produce exactly one leading tokenizer BOS token"
            )

    return policy


def validate_text_sequence_lengths(
    dataset: Any,
    *,
    processing_class: Any,
    policy: TextBoundaryPolicy,
    max_seq_length: int,
    source: str,
    batch_size: int = TEXT_VALIDATION_BATCH_SIZE,
) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("text validation batch size must be positive")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length <= 0
    ):
        raise ValueError("text max sequence length must be positive")

    token_limit = max_seq_length + 1
    effective_batch_size = min(
        batch_size,
        max(1, TEXT_VALIDATION_TOKEN_BUDGET // token_limit),
    )
    record_count = 0
    batch_start = 0
    normalized_texts: list[str] = []

    def validate_batch() -> None:
        input_id_rows = text_token_id_rows(
            processing_class,
            normalized_texts,
            add_special_tokens=policy.add_special_tokens,
            max_length=token_limit,
            description=(
                f"records {batch_start}-{batch_start + len(normalized_texts) - 1}"
            ),
        )
        for batch_index, input_ids in enumerate(input_id_rows):
            record_index = batch_start + batch_index
            subject = dataset_error_subject(
                DATASET_TYPE_TEXT,
                source=source,
                record_index=record_index,
            )
            if len(input_ids) > max_seq_length:
                raise ValueError(
                    f"{subject} produces at least {len(input_ids)} tokens after "
                    f"boundary normalization, exceeding maxSeqLength {max_seq_length}; "
                    "training truncation would discard part of the normalized record"
                )
            if trailing_token_count(input_ids, policy.eos_token_id) != 1:
                raise ValueError(
                    f"{subject} must have exactly one terminal tokenizer EOS token "
                    "after normalization"
                )
            if policy.bos_token_id is not None and leading_token_count(
                input_ids, policy.bos_token_id
            ) != 1:
                raise ValueError(
                    f"{subject} must have exactly one leading tokenizer BOS token "
                    "after normalization"
                )

    for record in dataset:
        normalized_texts.append(
            normalize_text_value(record["text"], policy=policy)
        )
        record_count += 1
        if len(normalized_texts) == effective_batch_size:
            validate_batch()
            normalized_texts.clear()
            batch_start = record_count

    if normalized_texts:
        validate_batch()


def prepare_training_dataset(
    dataset: Any,
    *,
    dataset_type: str,
    end_of_sequence: str,
    text_policy: TextBoundaryPolicy | None = None,
) -> Any:
    if dataset_type == DATASET_TYPE_ALPACA:
        return dataset.map(
            partial(format_alpaca_examples, end_of_sequence=end_of_sequence),
            batched=True,
        )
    if dataset_type in CHAT_DATASET_TYPES:
        return dataset
    if dataset_type == DATASET_TYPE_PROMPT_COMPLETION:
        return dataset
    if dataset_type == DATASET_TYPE_TEXT:
        if text_policy is None:
            raise RuntimeError("text preprocessing requires a boundary policy")
        return dataset.map(
            partial(normalize_text_examples, policy=text_policy),
            batched=True,
        )

    raise ValueError(f"unsupported dataset type {dataset_type!r}")


def canonical_string_examples(
    examples: Mapping[str, Sequence[str]],
    *,
    fields: Sequence[str],
) -> dict[str, list[str]]:
    return {field: list(examples[field]) for field in fields}


def normalize_canonical_string_dataset(
    dataset: Any,
    *,
    fields: Sequence[str],
    dataset_type: str,
    source: str,
    dataset_index: int,
) -> Any:
    subject = dataset_error_subject(
        dataset_type,
        source=source,
        dataset_index=dataset_index,
    )
    canonical_fields = tuple(fields)
    map_kwargs: dict[str, Any] = {}
    canonical_features = None
    if getattr(dataset, "features", None) is not None:
        try:
            from datasets import Features, Value

            canonical_features = Features(
                {field: Value("string") for field in canonical_fields}
            )
            map_kwargs["features"] = canonical_features
        except Exception:
            raise RuntimeError(
                f"{subject} could not construct canonical string features"
            ) from None

    try:
        normalized = dataset.map(
            partial(canonical_string_examples, fields=canonical_fields),
            batched=True,
            remove_columns=list(dataset.column_names),
            **map_kwargs,
        )
    except Exception:
        raise RuntimeError(
            f"{subject} could not be normalized to canonical string features"
        ) from None

    if list(getattr(normalized, "column_names", ())) != list(canonical_fields):
        raise RuntimeError(
            f"{subject} did not normalize to the canonical columns "
            f"{list(canonical_fields)!r}"
        )
    if (
        canonical_features is not None
        and getattr(normalized, "features", None) != canonical_features
    ):
        raise RuntimeError(
            f"{subject} did not normalize to canonical string features"
        )
    return normalized


def validate_full_sequence_text_tokenization(
    dataset: Any,
    *,
    processing_class: Any,
    max_seq_length: int,
    dataset_type: str,
    source: str,
    dataset_index: int,
    add_special_tokens: bool | None = None,
    batch_size: int = TEXT_VALIDATION_BATCH_SIZE,
) -> MessagesTokenFingerprint:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("full-sequence validation batch size must be positive")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length <= 0
    ):
        raise ValueError("full-sequence max sequence length must be positive")

    effective_batch_size = min(
        batch_size,
        max(1, TEXT_VALIDATION_TOKEN_BUDGET // max_seq_length),
    )
    fingerprint = empty_messages_token_fingerprint()
    texts: list[str] = []
    batch_start = 0
    record_count = 0
    effective_add_special_tokens = add_special_tokens
    source_add_special_tokens = None
    unsloth_tokenizer = getattr(
        processing_class,
        "tokenizer",
        processing_class,
    )

    def validate_batch() -> None:
        nonlocal fingerprint
        if source_add_special_tokens is None:
            raise RuntimeError(
                "full-sequence validation did not derive a source tokenization policy"
            )
        subject = dataset_error_subject(
            dataset_type,
            source=source,
            dataset_index=dataset_index,
        )
        input_id_rows = text_token_id_rows(
            unsloth_tokenizer,
            texts,
            add_special_tokens=bool(effective_add_special_tokens),
            max_length=max_seq_length,
            description=(
                f"{subject} rows {batch_start}-{batch_start + len(texts) - 1}"
            ),
        )
        source_input_id_rows = input_id_rows
        if source_add_special_tokens != effective_add_special_tokens:
            source_input_id_rows = text_token_id_rows(
                unsloth_tokenizer,
                texts,
                add_special_tokens=source_add_special_tokens,
                max_length=max_seq_length,
                description=(
                    f"{subject} source-policy rows {batch_start}-"
                    f"{batch_start + len(texts) - 1}"
                ),
            )
        for batch_index, (input_ids, source_input_ids) in enumerate(
            zip(input_id_rows, source_input_id_rows)
        ):
            record_index = batch_start + batch_index
            if input_ids != source_input_ids:
                row_subject = dataset_error_subject(
                    dataset_type,
                    source=source,
                    record_index=record_index,
                    dataset_index=dataset_index,
                )
                raise ValueError(
                    f"{row_subject} tokenizes differently with its source "
                    f"add_special_tokens={source_add_special_tokens} policy and "
                    "the combined full-sequence "
                    f"add_special_tokens={effective_add_special_tokens} policy "
                    "derived from datasets[0]; combining these records would "
                    "change tokenizer special-token boundaries"
                )
            if not input_ids:
                raise RuntimeError(
                    f"{subject} row {record_index} produced no training tokens"
                )
            fingerprint = extend_messages_token_fingerprint(
                fingerprint,
                input_ids,
                description=(
                    f"datasets[{dataset_index}] source record {record_index}"
                ),
            )

    for record in dataset:
        text = record["text"]
        if source_add_special_tokens is None:
            source_add_special_tokens = messages_unsloth_add_special_tokens(
                processing_class,
                text,
            )
        if effective_add_special_tokens is None:
            effective_add_special_tokens = source_add_special_tokens
        texts.append(text)
        record_count += 1
        if len(texts) == effective_batch_size:
            validate_batch()
            texts.clear()
            batch_start = record_count

    if texts:
        validate_batch()
    if record_count == 0:
        subject = dataset_error_subject(
            dataset_type,
            source=source,
            dataset_index=dataset_index,
        )
        raise RuntimeError(f"{subject} produced no canonical records")
    return fingerprint


def sequence_values(
    value: Any,
    *,
    description: str,
    preprocessing: str = "prompt-completion",
) -> list[Any]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(
            f"{preprocessing} preprocessing {description} must be a sequence"
        )

    return list(value)


def single_batch_row(
    value: Any,
    *,
    description: str,
    preprocessing: str = "prompt-completion",
) -> list[Any]:
    rows = sequence_values(
        value,
        description=description,
        preprocessing=preprocessing,
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"{preprocessing} preprocessing {description} must contain one row"
        )

    return sequence_values(
        rows[0],
        description=f"{description} row",
        preprocessing=preprocessing,
    )


def response_marker_token_ids(
    processing_class: Any,
    marker: str,
    *,
    description: str,
) -> tuple[int, ...]:
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    try:
        tokenized = tokenizer(marker, add_special_tokens=False)
    except Exception:
        raise RuntimeError(
            f"response-only messages preprocessing could not tokenize the {description}"
        ) from None
    if isinstance(tokenized, Mapping):
        try:
            raw_token_ids = tokenized["input_ids"]
        except KeyError as error:
            raise RuntimeError(
                f"response-only messages preprocessing {description} tokenization did not produce input_ids"
            ) from error
    else:
        raw_token_ids = getattr(tokenized, "input_ids", None)
        if raw_token_ids is None:
            raise RuntimeError(
                f"response-only messages preprocessing {description} tokenization did not produce input_ids"
            )

    token_ids = sequence_values(
        raw_token_ids,
        description=f"{description} token IDs",
        preprocessing="response-only messages",
    )
    if token_ids and isinstance(token_ids[0], Sequence):
        raise RuntimeError(
            f"response-only messages preprocessing {description} token IDs must be one-dimensional"
        )
    normalized_ids = []
    for token_index, raw_token_id in enumerate(token_ids):
        try:
            token_id = operator.index(raw_token_id)
        except TypeError as error:
            raise RuntimeError(
                f"response-only messages preprocessing {description} token ID {token_index} must be an integer"
            ) from error
        if isinstance(raw_token_id, bool) or token_id < 0:
            raise RuntimeError(
                f"response-only messages preprocessing {description} token ID {token_index} must be a non-negative integer"
            )
        normalized_ids.append(token_id)
    if not normalized_ids:
        raise RuntimeError(
            f"response-only messages preprocessing {description} must produce at least one token"
        )
    return tuple(normalized_ids)


def derive_response_markers(
    processing_class: Any,
    *,
    get_chat_template_parts: Callable[..., tuple[str, str]],
) -> ResponseMarkers:
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    use_tokenizer_parts = hasattr(
        tokenizer, "_unsloth_input_part"
    ) and hasattr(tokenizer, "_unsloth_output_part")
    if use_tokenizer_parts:
        parts = (
            getattr(tokenizer, "_unsloth_input_part"),
            getattr(tokenizer, "_unsloth_output_part"),
        )
    else:
        try:
            parts = get_chat_template_parts(processing_class)
        except Exception:
            raise RuntimeError(
                "response-only messages preprocessing could not derive stable instruction and response markers from the tokenizer chat template"
            ) from None
    if (
        not isinstance(parts, Sequence)
        or isinstance(parts, (str, bytes))
        or len(parts) != 2
    ):
        raise RuntimeError(
            "response-only messages preprocessing marker derivation must return exactly two markers"
        )
    instruction_part, response_part = parts
    if not isinstance(instruction_part, str) or not instruction_part.strip():
        raise RuntimeError(
            "response-only messages preprocessing instruction marker must be a non-empty string"
        )
    if not isinstance(response_part, str) or not response_part.strip():
        raise RuntimeError(
            "response-only messages preprocessing response marker must be a non-empty string"
        )
    if instruction_part == response_part:
        raise RuntimeError(
            "response-only messages preprocessing instruction and response markers must differ"
        )

    instruction_token_ids = response_marker_token_ids(
        processing_class,
        instruction_part,
        description="instruction marker",
    )
    response_token_ids = response_marker_token_ids(
        processing_class,
        response_part,
        description="response marker",
    )
    if instruction_token_ids == response_token_ids:
        raise RuntimeError(
            "response-only messages preprocessing instruction and response markers must tokenize differently"
        )
    if (
        token_subsequence_index(
            instruction_token_ids,
            response_token_ids,
            start=0,
        )
        is not None
        or token_subsequence_index(
            response_token_ids,
            instruction_token_ids,
            start=0,
        )
        is not None
    ):
        raise RuntimeError(
            "response-only messages preprocessing instruction and response marker token sequences must not contain one another"
        )
    return ResponseMarkers(
        instruction_part=instruction_part,
        response_part=response_part,
        instruction_token_ids=instruction_token_ids,
        response_token_ids=response_token_ids,
        use_tokenizer_parts=use_tokenizer_parts,
    )


def token_subsequence_index(
    token_ids: Sequence[Any],
    marker_ids: Sequence[int],
    *,
    start: int,
) -> int | None:
    last_start = len(token_ids) - len(marker_ids)
    for token_index in range(start, last_start + 1):
        if list(token_ids[token_index : token_index + len(marker_ids)]) == list(
            marker_ids
        ):
            return token_index
    return None


def token_subsequence_indices(
    token_ids: Sequence[Any],
    marker_ids: Sequence[int],
) -> list[int]:
    matches = []
    last_start = len(token_ids) - len(marker_ids)
    for token_index in range(last_start + 1):
        if list(token_ids[token_index : token_index + len(marker_ids)]) == list(
            marker_ids
        ):
            matches.append(token_index)
    return matches


def response_marker_events(
    input_ids: Sequence[Any],
    *,
    markers: ResponseMarkers,
    subject: str,
) -> list[tuple[int, str, int]]:
    events = [
        (token_index, "user", len(markers.instruction_token_ids))
        for token_index in token_subsequence_indices(
            input_ids,
            markers.instruction_token_ids,
        )
    ]
    events.extend(
        (token_index, "assistant", len(markers.response_token_ids))
        for token_index in token_subsequence_indices(
            input_ids,
            markers.response_token_ids,
        )
    )
    events.sort(key=lambda event: (event[0], event[1]))

    previous_end = 0
    for event_index, (token_index, _, marker_length) in enumerate(events):
        if event_index > 0 and token_index < previous_end:
            raise RuntimeError(
                f"{subject} has overlapping instruction and response marker token matches"
            )
        previous_end = token_index + marker_length
    return events


def validate_response_message_sequence(
    messages: Sequence[Mapping[str, str]],
    *,
    subject: str,
) -> None:
    expected_role = "user"
    saw_conversation_turn = False
    for message_index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            if saw_conversation_turn:
                raise ValueError(
                    f"{subject} response-only loss requires system messages "
                    "to precede all user and assistant messages"
                )
            continue

        saw_conversation_turn = True
        if role != expected_role:
            raise ValueError(
                f"{subject} response-only loss requires user and assistant "
                f"messages to alternate after any system prefix; message "
                f"{message_index} must have role {expected_role!r}"
            )
        expected_role = "assistant" if role == "user" else "user"

    if not saw_conversation_turn or expected_role != "user":
        raise ValueError(
            f"{subject} response-only loss requires one or more user and "
            "assistant pairs ending with an assistant message"
        )


def validate_response_training_dataset(
    dataset: Any,
    *,
    dataset_type: str,
    source: str,
) -> None:
    for record_index, record in enumerate(dataset):
        subject = dataset_error_subject(
            dataset_type,
            source=source,
            record_index=record_index,
        )
        validate_response_message_sequence(
            record["messages"],
            subject=subject,
        )


def validate_response_marker_layout(
    messages: Sequence[Mapping[str, str]],
    input_ids: Sequence[Any],
    *,
    markers: ResponseMarkers,
    subject: str,
) -> None:
    validate_response_message_sequence(messages, subject=subject)
    expected_roles = []
    for message_index, message in enumerate(messages):
        content = message["content"]
        if (
            markers.instruction_part in content
            or markers.response_part in content
        ):
            raise ValueError(
                f"{subject} message {message_index} content collides with "
                "response-only chat-template markers"
            )
        role = message["role"]
        if role == "user":
            expected_roles.append("user")
        elif role == "assistant":
            expected_roles.append("assistant")

    events = response_marker_events(
        input_ids,
        markers=markers,
        subject=f"{subject} response-only marker layout",
    )
    actual_roles = [role for _, role, _ in events]
    if actual_roles != expected_roles:
        raise ValueError(
            f"{subject} response-only chat-template markers do not uniquely "
            "match the user and assistant message boundaries; marker collision "
            "or unstable marker tokenization detected"
        )


def expected_response_only_labels(
    input_ids: Sequence[Any],
    *,
    markers: ResponseMarkers,
) -> list[Any]:
    """Reproduce the pinned force-match masking for one conversation."""
    labels = [-100] * len(input_ids)
    token_count = len(input_ids)
    last_token_index = token_count - 1
    token_index = 0
    response_spans = []
    while token_index < token_count:
        if list(
            input_ids[
                token_index : token_index + len(markers.response_token_ids)
            ]
        ) == list(markers.response_token_ids):
            response_start = token_index + len(markers.response_token_ids)
            token_index = response_start
            while token_index < token_count:
                instruction_start = token_index
                if token_index == last_token_index or list(
                    input_ids[
                        token_index : token_index
                        + len(markers.instruction_token_ids)
                    ]
                ) == list(markers.instruction_token_ids):
                    if token_index == last_token_index:
                        response_end = token_count
                        token_index = token_count
                    else:
                        response_end = instruction_start
                        token_index += len(markers.instruction_token_ids)
                    response_spans.append((response_start, response_end))
                    break
                token_index += 1
        token_index += 1

    for response_start, response_end in response_spans:
        labels[response_start:response_end] = input_ids[
            response_start:response_end
        ]
    return labels


@contextmanager
def prompt_completion_right_truncation(processing_class: Any) -> Iterator[None]:
    """Match the pinned trainer's prefix truncation without unbounded outputs."""
    candidates = [processing_class]
    inner_tokenizer = getattr(processing_class, "tokenizer", None)
    if inner_tokenizer is not None and inner_tokenizer is not processing_class:
        candidates.append(inner_tokenizer)

    changed_targets = []
    try:
        for candidate in candidates:
            truncation_side = getattr(candidate, "truncation_side", None)
            if truncation_side == "left":
                candidate.truncation_side = "right"
                changed_targets.append((candidate, truncation_side))
    except Exception:
        for candidate, truncation_side in reversed(changed_targets):
            candidate.truncation_side = truncation_side
        raise RuntimeError(
            "prompt-completion preprocessing could not enforce right-side token truncation"
        ) from None

    try:
        yield
    finally:
        for candidate, truncation_side in reversed(changed_targets):
            candidate.truncation_side = truncation_side


def tokenize_verification_text(
    processing_class: Any,
    text: str,
    *,
    add_special_tokens: bool,
    max_length: int,
    description: str,
) -> list[Any]:
    with prompt_completion_right_truncation(processing_class):
        tokenized = processing_class(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    if not isinstance(tokenized, Mapping):
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization must return a mapping"
        )

    try:
        input_ids = sequence_values(
            tokenized["input_ids"],
            description=f"{description} token IDs",
        )
    except KeyError as error:
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization did not produce input_ids"
        ) from error
    if len(input_ids) > max_length:
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization did not honor its truncation limit"
        )
    return input_ids


def prompt_completion_add_special_tokens(
    processing_class: Any,
    prompt: str,
) -> bool:
    """Match the active Unsloth prompt-completion special-token decision."""
    bos_token = getattr(processing_class, "bos_token", None)
    chat_template = getattr(processing_class, "chat_template", "") or ""
    return not (
        bos_token is not None
        and (prompt.startswith(bos_token) or bos_token in chat_template)
    )


def tokenize_verification_texts(
    processing_class: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
    max_length: int,
    description: str,
) -> list[list[Any]]:
    with prompt_completion_right_truncation(processing_class):
        tokenized = processing_class(
            list(texts),
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    if not isinstance(tokenized, Mapping):
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization must return a mapping"
        )

    try:
        token_rows = sequence_values(
            tokenized["input_ids"],
            description=f"{description} token ID rows",
        )
    except KeyError as error:
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization did not produce input_ids"
        ) from error

    if len(token_rows) != len(texts):
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization returned an unexpected number of rows"
        )

    input_id_rows = []
    for row_index, token_row in enumerate(token_rows):
        input_ids = sequence_values(
            token_row,
            description=f"{description} token IDs row {row_index}",
        )
        if len(input_ids) > max_length:
            raise RuntimeError(
                f"prompt-completion preprocessing {description} tokenization did not honor its truncation limit"
            )
        input_id_rows.append(input_ids)
    return input_id_rows


def empty_prompt_prefix_fingerprint() -> PromptPrefixFingerprint:
    return PromptPrefixFingerprint(0, 0, 0)


def merge_prompt_prefix_fingerprints(
    fingerprints: Sequence[PromptPrefixFingerprint],
) -> PromptPrefixFingerprint:
    merged = empty_prompt_prefix_fingerprint()
    for fingerprint in fingerprints:
        merged = PromptPrefixFingerprint(
            merged.sequence_count + fingerprint.sequence_count,
            (merged.first_digest_sum + fingerprint.first_digest_sum)
            & PROMPT_PREFIX_FINGERPRINT_MASK,
            (merged.second_digest_sum + fingerprint.second_digest_sum)
            & PROMPT_PREFIX_FINGERPRINT_MASK,
        )
    return merged


def extend_prompt_prefix_fingerprint(
    fingerprint: PromptPrefixFingerprint,
    token_ids: Sequence[Any],
    *,
    description: str,
) -> PromptPrefixFingerprint:
    encoded_token_ids = bytearray()
    for token_index, raw_token_id in enumerate(token_ids):
        try:
            token_id = operator.index(raw_token_id)
        except TypeError as error:
            raise RuntimeError(
                f"prompt-completion preprocessing {description} token ID {token_index} must be an integer"
            ) from error
        if isinstance(raw_token_id, bool) or not 0 <= token_id < 1 << 64:
            raise RuntimeError(
                f"prompt-completion preprocessing {description} token ID {token_index} must be an unsigned 64-bit integer"
            )
        encoded_token_ids.extend(token_id.to_bytes(8, byteorder="big"))

    first_hasher = hashlib.sha256(b"aikit-prompt-prefix-1\x00")
    first_hasher.update(encoded_token_ids)
    second_hasher = hashlib.sha256(b"aikit-prompt-prefix-2\x00")
    second_hasher.update(encoded_token_ids)
    first_digest = int.from_bytes(first_hasher.digest(), byteorder="big")
    second_digest = int.from_bytes(second_hasher.digest(), byteorder="big")

    return PromptPrefixFingerprint(
        fingerprint.sequence_count + 1,
        (fingerprint.first_digest_sum + first_digest)
        & PROMPT_PREFIX_FINGERPRINT_MASK,
        (fingerprint.second_digest_sum + second_digest)
        & PROMPT_PREFIX_FINGERPRINT_MASK,
    )


def validate_prompt_completion_tokenization(
    dataset: Any,
    *,
    processing_class: Any,
    max_seq_length: int,
    add_special_tokens: bool | None = None,
    batch_size: int = PROMPT_COMPLETION_VALIDATION_BATCH_SIZE,
) -> PromptPrefixFingerprint:
    """Validate source token boundaries in bounded tokenizer batches."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("prompt-completion validation batch size must be positive")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length <= 0
    ):
        raise ValueError("prompt-completion max sequence length must be positive")

    estimated_tokens_per_record = max_seq_length * 2
    effective_batch_size = min(
        batch_size,
        max(
            1,
            PROMPT_COMPLETION_VALIDATION_TOKEN_BUDGET
            // estimated_tokens_per_record,
        ),
    )

    eos_token = getattr(processing_class, "eos_token", None)
    eos_token_id = getattr(processing_class, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError(
            "prompt-completion preprocessing requires a non-empty tokenizer EOS token"
        )
    if not isinstance(eos_token_id, int):
        raise RuntimeError(
            "prompt-completion preprocessing requires an integer tokenizer EOS token ID"
        )

    record_count = 0
    batch_start = 0
    prompts: list[str] = []
    prompt_completions: list[str] = []
    effective_add_special_tokens = add_special_tokens
    source_add_special_tokens = None
    fingerprint = empty_prompt_prefix_fingerprint()

    def validate_batch() -> None:
        nonlocal fingerprint
        if source_add_special_tokens is None:
            raise RuntimeError(
                "prompt-completion validation did not derive a source tokenization policy"
            )
        token_rows = tokenize_verification_texts(
            processing_class,
            prompts + prompt_completions,
            add_special_tokens=effective_add_special_tokens,
            max_length=max_seq_length,
            description=(
                f"records {batch_start}-{batch_start + len(prompts) - 1} "
                "prompts and prompt-completions"
            ),
        )
        prompt_token_rows = token_rows[: len(prompts)]
        prompt_completion_token_rows = token_rows[len(prompts) :]
        source_prompt_token_rows = prompt_token_rows
        source_prompt_completion_token_rows = prompt_completion_token_rows
        if source_add_special_tokens != effective_add_special_tokens:
            source_token_rows = tokenize_verification_texts(
                processing_class,
                prompts + prompt_completions,
                add_special_tokens=source_add_special_tokens,
                max_length=max_seq_length,
                description=(
                    f"source-policy records {batch_start}-"
                    f"{batch_start + len(prompts) - 1} prompts and prompt-completions"
                ),
            )
            source_prompt_token_rows = source_token_rows[: len(prompts)]
            source_prompt_completion_token_rows = source_token_rows[len(prompts) :]
        for batch_index, (
            prompt_ids,
            input_ids,
            source_prompt_ids,
            source_input_ids,
        ) in enumerate(
            zip(
                prompt_token_rows,
                prompt_completion_token_rows,
                source_prompt_token_rows,
                source_prompt_completion_token_rows,
            )
        ):
            record_index = batch_start + batch_index
            prompt_length = min(len(prompt_ids), len(input_ids))
            source_prompt_length = min(
                len(source_prompt_ids),
                len(source_input_ids),
            )
            completion_mask = [0] * prompt_length + [1] * (
                len(input_ids) - prompt_length
            )
            source_completion_mask = [0] * source_prompt_length + [1] * (
                len(source_input_ids) - source_prompt_length
            )
            if (
                input_ids != source_input_ids
                or completion_mask != source_completion_mask
            ):
                raise ValueError(
                    f"prompt-completion preprocessing record {record_index} "
                    "tokenizes differently with its source "
                    f"add_special_tokens={source_add_special_tokens} policy and "
                    "the combined dataset "
                    f"add_special_tokens={effective_add_special_tokens} policy; "
                    "combining these records would change token or completion-mask "
                    "boundaries"
                )
            if len(input_ids) <= prompt_length:
                raise RuntimeError(
                    f"prompt-completion preprocessing record {record_index} retains no completion tokens after truncation to maxSeqLength {max_seq_length}"
                )
            if input_ids[-1] != eos_token_id:
                raise RuntimeError(
                    f"prompt-completion preprocessing record {record_index} does not end with a supervised EOS token after truncation to maxSeqLength {max_seq_length}"
                )
            fingerprint = extend_prompt_prefix_fingerprint(
                fingerprint,
                input_ids[:prompt_length],
                description=f"record {record_index} prompt",
            )

    for record in dataset:
        if source_add_special_tokens is None:
            source_add_special_tokens = prompt_completion_add_special_tokens(
                processing_class,
                record["prompt"],
            )
        if effective_add_special_tokens is None:
            effective_add_special_tokens = source_add_special_tokens

        completion = record["completion"]
        if not completion.endswith(eos_token):
            completion += eos_token
        prompts.append(record["prompt"])
        prompt_completions.append(record["prompt"] + completion)
        record_count += 1

        if len(prompts) == effective_batch_size:
            validate_batch()
            prompts.clear()
            prompt_completions.clear()
            batch_start = record_count

    if prompts:
        validate_batch()
    if record_count == 0:
        raise RuntimeError(
            "prompt-completion preprocessing produced an empty source dataset"
        )
    return fingerprint


def verify_prompt_completion_preprocessing(
    trainer: Any,
    *,
    dataset_from_dict: Callable[..., Any],
    processing_class: Any,
) -> None:
    """Verify the active Unsloth/TRL preprocessing and label-masking contract."""
    eos_token = getattr(processing_class, "eos_token", None)
    eos_token_id = getattr(processing_class, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError(
            "prompt-completion preprocessing requires a non-empty tokenizer EOS token"
        )
    if not isinstance(eos_token_id, int):
        raise RuntimeError(
            "prompt-completion preprocessing requires an integer tokenizer EOS token ID"
        )

    add_special_tokens = prompt_completion_add_special_tokens(
        processing_class,
        PREPROCESSING_VERIFICATION_PROMPT,
    )
    expected_prompt_ids = tokenize_verification_text(
        processing_class,
        PREPROCESSING_VERIFICATION_PROMPT,
        add_special_tokens=add_special_tokens,
        max_length=PROMPT_COMPLETION_PREPROCESSING_VERIFICATION_MAX_LENGTH,
        description="prompt",
    )
    verification_completion = PREPROCESSING_VERIFICATION_COMPLETION
    if not verification_completion.endswith(eos_token):
        verification_completion += eos_token
    expected_input_ids = tokenize_verification_text(
        processing_class,
        PREPROCESSING_VERIFICATION_PROMPT + verification_completion,
        add_special_tokens=add_special_tokens,
        max_length=PROMPT_COMPLETION_PREPROCESSING_VERIFICATION_MAX_LENGTH,
        description="prompt-completion",
    )
    if not expected_prompt_ids or not expected_input_ids:
        raise RuntimeError(
            "prompt-completion preprocessing verification text must produce tokens"
        )

    verification_args = copy.copy(trainer.args)
    verification_args.max_length = len(expected_input_ids)
    verification_args.dataset_num_proc = 1
    verification_dataset = dataset_from_dict(
        {
            "prompt": [PREPROCESSING_VERIFICATION_PROMPT],
            "completion": [PREPROCESSING_VERIFICATION_COMPLETION],
        }
    )
    prepared_dataset = trainer._prepare_dataset(
        verification_dataset,
        processing_class,
        verification_args,
        bool(getattr(trainer.args, "packing", False)),
        None,
        "prompt-completion verification",
    )
    prepared_record = prepared_dataset[0]
    if not isinstance(prepared_record, Mapping):
        raise RuntimeError(
            "prompt-completion preprocessing must produce mapping records"
        )

    try:
        input_ids = sequence_values(
            prepared_record["input_ids"],
            description="input_ids",
        )
        completion_mask = sequence_values(
            prepared_record["completion_mask"],
            description="completion_mask",
        )
    except KeyError as error:
        raise RuntimeError(
            f"prompt-completion preprocessing did not produce {error.args[0]}"
        ) from error

    if len(input_ids) != len(completion_mask):
        raise RuntimeError(
            "prompt-completion preprocessing input_ids and completion_mask lengths differ"
        )
    if 0 not in completion_mask or 1 not in completion_mask:
        raise RuntimeError(
            "prompt-completion preprocessing must identify prompt and completion tokens"
        )

    first_completion = completion_mask.index(1)
    if first_completion != len(expected_prompt_ids):
        raise RuntimeError(
            "prompt-completion preprocessing completion mask boundary does not match the tokenized prompt"
        )
    if completion_mask[:first_completion] != [0] * first_completion or completion_mask[
        first_completion:
    ] != [1] * (len(completion_mask) - first_completion):
        raise RuntimeError(
            "prompt-completion preprocessing must mask a prompt prefix and completion suffix"
        )

    collated = trainer.data_collator([prepared_record])
    if not isinstance(collated, Mapping):
        raise RuntimeError(
            "prompt-completion preprocessing data collator must return a mapping"
        )
    try:
        collated_input_ids = single_batch_row(
            collated["input_ids"],
            description="collated input_ids",
        )
        labels = single_batch_row(
            collated["labels"],
            description="collated labels",
        )
    except KeyError as error:
        raise RuntimeError(
            f"prompt-completion preprocessing data collator did not produce {error.args[0]}"
        ) from error

    if collated_input_ids[: len(input_ids)] != input_ids:
        raise RuntimeError(
            "prompt-completion preprocessing data collator changed the token sequence"
        )
    if len(labels) < len(input_ids):
        raise RuntimeError(
            "prompt-completion preprocessing labels are shorter than input_ids"
        )

    for index, mask_value in enumerate(completion_mask):
        if mask_value == 0 and labels[index] != -100:
            raise RuntimeError(
                "prompt-completion preprocessing prompt tokens must be masked"
            )
        if mask_value == 1 and labels[index] != input_ids[index]:
            raise RuntimeError(
                "prompt-completion preprocessing completion tokens must be supervised"
            )

    if input_ids[-1] != eos_token_id:
        raise RuntimeError(
            "prompt-completion preprocessing must end with the tokenizer EOS token"
        )
    if completion_mask[-1] != 1 or labels[len(input_ids) - 1] != eos_token_id:
        raise RuntimeError(
            "prompt-completion preprocessing EOS token must be supervised"
        )


def validate_prepared_prompt_completion_dataset(
    dataset: Any,
    *,
    eos_token_id: int,
    max_seq_length: int,
    packing: bool,
    packing_strategy: str = "bfd",
    expected_prompt_prefix_fingerprint: PromptPrefixFingerprint | None = None,
) -> None:
    """Validate completion supervision after real-record preprocessing."""
    if packing and packing_strategy != "bfd":
        raise RuntimeError(
            "prompt-completion preprocessing validation requires the bfd packing strategy"
        )

    record_count = 0
    prompt_prefix_fingerprint = empty_prompt_prefix_fingerprint()
    for record_index, record in enumerate(dataset):
        record_count += 1
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"prompt-completion preprocessing record {record_index} must be a mapping"
            )

        try:
            input_ids = sequence_values(
                record["input_ids"],
                description=f"record {record_index} input_ids",
            )
            completion_mask = sequence_values(
                record["completion_mask"],
                description=f"record {record_index} completion_mask",
            )
        except KeyError as error:
            raise RuntimeError(
                f"prompt-completion preprocessing record {record_index} did not produce {error.args[0]}"
            ) from error

        if not input_ids:
            raise RuntimeError(
                f"prompt-completion preprocessing record {record_index} produced no tokens"
            )
        if len(input_ids) != len(completion_mask):
            raise RuntimeError(
                f"prompt-completion preprocessing record {record_index} input_ids and completion_mask lengths differ"
            )

        if packing:
            try:
                raw_sequence_lengths = sequence_values(
                    record["seq_lengths"],
                    description=f"record {record_index} seq_lengths",
                )
            except KeyError as error:
                raise RuntimeError(
                    f"prompt-completion preprocessing packed record {record_index} did not produce seq_lengths"
                ) from error

            sequence_lengths = []
            for raw_sequence_length in raw_sequence_lengths:
                try:
                    sequence_length = int(raw_sequence_length)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"prompt-completion preprocessing packed record {record_index} has an invalid sequence length"
                    ) from error
                if (
                    isinstance(raw_sequence_length, bool)
                    or sequence_length != raw_sequence_length
                    or sequence_length <= 0
                ):
                    raise RuntimeError(
                        f"prompt-completion preprocessing packed record {record_index} has an invalid sequence length"
                    )
                sequence_lengths.append(sequence_length)

            if not sequence_lengths or sum(sequence_lengths) != len(input_ids):
                raise RuntimeError(
                    f"prompt-completion preprocessing packed record {record_index} seq_lengths do not match input_ids"
                )
        else:
            sequence_lengths = [len(input_ids)]

        offset = 0
        for segment_index, sequence_length in enumerate(sequence_lengths):
            end = offset + sequence_length
            sequence_input_ids = input_ids[offset:end]
            sequence_completion_mask = completion_mask[offset:end]
            location = f"record {record_index}"
            if packing:
                location += f" packed segment {segment_index}"

            if 1 not in sequence_completion_mask:
                raise RuntimeError(
                    f"prompt-completion preprocessing {location} retains no completion tokens after truncation to maxSeqLength {max_seq_length}"
                )
            first_completion = sequence_completion_mask.index(1)
            if sequence_completion_mask != [0] * first_completion + [1] * (
                len(sequence_completion_mask) - first_completion
            ):
                raise RuntimeError(
                    f"prompt-completion preprocessing {location} must mask a prompt prefix and completion suffix"
                )
            prompt_prefix_fingerprint = extend_prompt_prefix_fingerprint(
                prompt_prefix_fingerprint,
                sequence_input_ids[:first_completion],
                description=f"{location} prompt",
            )
            if (
                sequence_input_ids[-1] != eos_token_id
                or sequence_completion_mask[-1] != 1
            ):
                raise RuntimeError(
                    f"prompt-completion preprocessing {location} does not end with a supervised EOS token after truncation to maxSeqLength {max_seq_length}"
                )
            offset = end

    if record_count == 0:
        raise RuntimeError(
            "prompt-completion preprocessing produced an empty training dataset"
        )
    if (
        expected_prompt_prefix_fingerprint is not None
        and prompt_prefix_fingerprint != expected_prompt_prefix_fingerprint
    ):
        raise RuntimeError(
            "prompt-completion preprocessing prepared prompt prefixes do not match the source prompts"
        )


def messages_preprocessing_segments(
    prepared_record: Mapping[str, Any],
    *,
    record_index: int,
    packing: bool,
) -> tuple[list[Any], list[list[Any]]]:
    try:
        input_ids = sequence_values(
            prepared_record["input_ids"],
            description=f"record {record_index} input_ids",
            preprocessing="messages",
        )
    except KeyError as error:
        raise RuntimeError(
            f"messages preprocessing record {record_index} did not produce input_ids"
        ) from error
    if not input_ids:
        raise RuntimeError(
            f"messages preprocessing record {record_index} produced no tokens"
        )
    if not packing:
        return input_ids, [input_ids]

    try:
        raw_sequence_lengths = sequence_values(
            prepared_record["seq_lengths"],
            description=f"record {record_index} seq_lengths",
            preprocessing="messages",
        )
    except KeyError as error:
        raise RuntimeError(
            f"messages preprocessing packed record {record_index} did not produce seq_lengths"
        ) from error

    sequence_lengths = []
    for raw_sequence_length in raw_sequence_lengths:
        try:
            sequence_length = operator.index(raw_sequence_length)
        except TypeError as error:
            raise RuntimeError(
                f"messages preprocessing packed record {record_index} has an invalid sequence length"
            ) from error
        if isinstance(raw_sequence_length, bool) or sequence_length <= 0:
            raise RuntimeError(
                f"messages preprocessing packed record {record_index} has an invalid sequence length"
            )
        sequence_lengths.append(sequence_length)
    if not sequence_lengths or sum(sequence_lengths) != len(input_ids):
        raise RuntimeError(
            f"messages preprocessing packed record {record_index} seq_lengths do not match input_ids"
        )

    segments = []
    offset = 0
    for sequence_length in sequence_lengths:
        segments.append(input_ids[offset : offset + sequence_length])
        offset += sequence_length
    return input_ids, segments


def validate_prepared_messages_dataset(
    dataset: Any,
    *,
    data_collator: Callable[[list[Mapping[str, Any]]], Mapping[str, Any]],
    max_seq_length: int,
    packing: bool,
    padding_free: bool,
    packing_strategy: str = "bfd",
    expected_fingerprint: MessagesTokenFingerprint,
    loss: str = LOSS_ALL,
    response_markers: ResponseMarkers | None = None,
) -> None:
    """Verify actual rendered-message boundaries and loss labels."""
    if loss == LOSS_RESPONSE and packing:
        raise RuntimeError(
            "response-only messages preprocessing does not support packing "
            "because response masks must not cross conversation boundaries"
        )
    if packing and packing_strategy != "bfd":
        raise RuntimeError(
            "messages preprocessing validation requires the bfd packing strategy"
        )
    if loss not in SUPPORTED_LOSSES:
        raise RuntimeError(f"messages preprocessing has unsupported loss {loss!r}")
    if loss == LOSS_RESPONSE and response_markers is None:
        raise RuntimeError(
            "response-only messages preprocessing requires validated markers"
        )
    if loss == LOSS_ALL and response_markers is not None:
        raise RuntimeError(
            "full-sequence messages preprocessing must not use response markers"
        )

    fingerprint = empty_messages_token_fingerprint()
    record_count = 0
    supervised_response_tokens = 0
    for record_index, prepared_record in enumerate(dataset):
        record_count += 1
        if not isinstance(prepared_record, Mapping):
            raise RuntimeError(
                f"messages preprocessing record {record_index} must be a mapping"
            )
        for mask_field in ("assistant_masks", "completion_mask"):
            if mask_field in prepared_record:
                raise RuntimeError(
                    f"messages preprocessing must not produce {mask_field}"
                )

        input_ids, segments = messages_preprocessing_segments(
            prepared_record,
            record_index=record_index,
            packing=packing,
        )
        if len(input_ids) > max_seq_length:
            raise RuntimeError(
                f"messages preprocessing record {record_index} exceeds maxSeqLength {max_seq_length}"
            )

        prepared_labels = None
        if loss == LOSS_RESPONSE:
            try:
                prepared_labels = sequence_values(
                    prepared_record["labels"],
                    description=f"record {record_index} labels",
                    preprocessing="response-only messages",
                )
            except KeyError as error:
                raise RuntimeError(
                    f"response-only messages preprocessing record {record_index} did not produce labels"
                ) from error
            if len(prepared_labels) != len(input_ids):
                raise RuntimeError(
                    f"response-only messages preprocessing record {record_index} input_ids and labels lengths differ"
                )

        sequence_starts = set()
        sequence_ends = set()
        offset = 0
        for segment_index, segment in enumerate(segments):
            location = f"prepared record {record_index}"
            if packing:
                location += f" packed segment {segment_index}"
            if not segment or len(segment) > max_seq_length:
                raise RuntimeError(
                    f"messages preprocessing {location} has an invalid sequence length"
            )
            sequence_starts.add(offset)
            offset += len(segment)
            sequence_ends.add(offset - 1)
            fingerprint = extend_messages_token_fingerprint(
                fingerprint,
                segment,
                description=location,
            )

            if loss == LOSS_RESPONSE:
                expected_labels = expected_response_only_labels(
                    segment,
                    markers=response_markers,
                )
                if all(label == -100 for label in expected_labels):
                    raise RuntimeError(
                        f"response-only messages preprocessing {location} has no supervised response tokens"
                    )
                actual_labels = prepared_labels[
                    offset - len(segment) : offset
                ]
                for expected_label, actual_label in zip(
                    expected_labels, actual_labels
                ):
                    if actual_label == expected_label:
                        continue
                    if expected_label == -100:
                        raise RuntimeError(
                            f"response-only messages preprocessing {location} must mask all non-response tokens"
                        )
                    raise RuntimeError(
                        f"response-only messages preprocessing {location} assistant response tokens must be supervised"
                    )
                supervised_response_tokens += sum(
                    label != -100 for label in actual_labels
                )

        collated = data_collator([prepared_record])
        if not isinstance(collated, Mapping):
            raise RuntimeError(
                "messages preprocessing data collator must return a mapping"
            )
        for mask_field in ("assistant_masks", "completion_mask"):
            if mask_field in collated:
                raise RuntimeError(
                    f"messages preprocessing data collator must not produce {mask_field}"
                )
        try:
            collated_input_ids = single_batch_row(
                collated["input_ids"],
                description="collated input_ids",
                preprocessing="messages",
            )
            labels = single_batch_row(
                collated["labels"],
                description="collated labels",
                preprocessing="messages",
            )
        except KeyError as error:
            raise RuntimeError(
                f"messages preprocessing data collator did not produce {error.args[0]}"
            ) from error
        if collated_input_ids[: len(input_ids)] != input_ids:
            raise RuntimeError(
                "messages preprocessing data collator changed the token sequence"
            )
        if len(labels) < len(input_ids):
            raise RuntimeError(
                "messages preprocessing labels are shorter than input_ids"
            )

        if loss == LOSS_RESPONSE:
            for token_index, prepared_label in enumerate(prepared_labels):
                if labels[token_index] == prepared_label:
                    continue
                if prepared_label == -100:
                    raise RuntimeError(
                        "response-only messages preprocessing data collator unmasked non-response tokens"
                    )
                raise RuntimeError(
                    "response-only messages preprocessing data collator masked assistant response tokens"
                )
        else:
            allow_sequence_start_masking = packing or padding_free
            for token_index, input_id in enumerate(input_ids):
                if labels[token_index] == input_id:
                    continue
                if (
                    allow_sequence_start_masking
                    and token_index in sequence_starts
                    and labels[token_index] == -100
                ):
                    continue
                raise RuntimeError(
                    "messages preprocessing must use full-sequence labels"
                )
            for sequence_end in sequence_ends:
                if labels[sequence_end] != input_ids[sequence_end]:
                    raise RuntimeError(
                        "messages preprocessing final tokens must be supervised"
                    )

    if record_count == 0:
        raise RuntimeError(
            "messages preprocessing produced an empty training dataset"
        )
    if loss == LOSS_RESPONSE and supervised_response_tokens == 0:
        raise RuntimeError(
            "response-only messages preprocessing training dataset has no supervised response tokens"
        )
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            "messages preprocessing prepared token sequences do not match the canonical conversations"
        )


def text_preprocessing_segments(
    prepared_record: Mapping[str, Any],
    *,
    packing: bool,
) -> tuple[list[Any], list[list[Any]]]:
    try:
        input_ids = sequence_values(
            prepared_record["input_ids"],
            description="input_ids",
            preprocessing="text",
        )
    except KeyError as error:
        raise RuntimeError(
            "text preprocessing did not produce input_ids"
        ) from error

    if not packing:
        return input_ids, [input_ids]

    try:
        sequence_lengths = sequence_values(
            prepared_record["seq_lengths"],
            description="seq_lengths",
            preprocessing="text",
        )
    except KeyError as error:
        raise RuntimeError(
            "text preprocessing packing did not preserve sequence lengths"
        ) from error
    if (
        not sequence_lengths
        or any(
            not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            for length in sequence_lengths
        )
        or sum(sequence_lengths) != len(input_ids)
    ):
        raise RuntimeError(
            "text preprocessing packed sequence lengths do not match input_ids"
        )

    segments = []
    offset = 0
    for sequence_length in sequence_lengths:
        segments.append(input_ids[offset : offset + sequence_length])
        offset += sequence_length
    return input_ids, segments


def verify_text_preprocessing(
    trainer: Any,
    *,
    dataset_from_dict: Callable[..., Any],
    processing_class: Any,
    policy: TextBoundaryPolicy,
) -> None:
    """Verify the active Unsloth/TRL full-sequence text contract."""
    normalized_texts = [
        normalize_text_value(text, policy=policy)
        for text in text_preprocessing_verification_sources(policy)
    ]
    expected_segments = [
        text_token_ids(
            processing_class,
            text,
            add_special_tokens=policy.add_special_tokens,
            description=f"verification row {index}",
        )
        for index, text in enumerate(normalized_texts)
    ]
    if any(not input_ids for input_ids in expected_segments):
        raise RuntimeError(
            "text preprocessing verification text must produce tokens"
        )

    packing = bool(getattr(trainer.args, "packing", False))
    verification_args = copy.copy(trainer.args)
    verification_args.max_length = sum(map(len, expected_segments))
    verification_args.dataset_num_proc = 1
    verification_dataset = dataset_from_dict({"text": normalized_texts})
    prepared_dataset = trainer._prepare_dataset(
        verification_dataset,
        processing_class,
        verification_args,
        packing,
        None,
        "text verification",
    )

    prepared_records = list(prepared_dataset)
    if not prepared_records:
        raise RuntimeError("text preprocessing produced no verification records")

    actual_segments = []
    record_details = []
    for prepared_record in prepared_records:
        if not isinstance(prepared_record, Mapping):
            raise RuntimeError(
                "text preprocessing must produce mapping records"
            )
        input_ids, segments = text_preprocessing_segments(
            prepared_record,
            packing=packing,
        )
        actual_segments.extend(segments)
        record_details.append((prepared_record, input_ids, segments))

    for segment in actual_segments:
        if trailing_token_count(segment, policy.eos_token_id) != 1:
            raise RuntimeError(
                "text preprocessing must retain exactly one terminal EOS per record"
            )
        if policy.bos_token_id is not None and leading_token_count(
            segment, policy.bos_token_id
        ) != 1:
            raise RuntimeError(
                "text preprocessing must retain exactly one leading BOS per record"
            )

    unmatched_expected = [list(segment) for segment in expected_segments]
    for segment in actual_segments:
        try:
            unmatched_expected.remove(segment)
        except ValueError as error:
            raise RuntimeError(
                "text preprocessing changed a normalized token sequence"
            ) from error
    if unmatched_expected:
        raise RuntimeError(
            "text preprocessing omitted a normalized token sequence"
        )

    for prepared_record, input_ids, segments in record_details:
        collated = trainer.data_collator([prepared_record])
        if not isinstance(collated, Mapping):
            raise RuntimeError(
                "text preprocessing data collator must return a mapping"
            )
        try:
            collated_input_ids = single_batch_row(
                collated["input_ids"],
                description="collated input_ids",
                preprocessing="text",
            )
            labels = single_batch_row(
                collated["labels"],
                description="collated labels",
                preprocessing="text",
            )
        except KeyError as error:
            raise RuntimeError(
                f"text preprocessing data collator did not produce {error.args[0]}"
            ) from error

        if collated_input_ids[: len(input_ids)] != input_ids:
            raise RuntimeError(
                "text preprocessing data collator changed the token sequence"
            )
        if len(labels) < len(input_ids):
            raise RuntimeError(
                "text preprocessing labels are shorter than input_ids"
            )

        sequence_starts = set()
        sequence_ends = set()
        offset = 0
        for segment in segments:
            sequence_starts.add(offset)
            offset += len(segment)
            sequence_ends.add(offset - 1)

        allow_sequence_start_masking = packing or bool(
            getattr(trainer.args, "padding_free", False)
        )
        for index, input_id in enumerate(input_ids):
            if labels[index] == input_id:
                continue
            if (
                allow_sequence_start_masking
                and index in sequence_starts
                and labels[index] == -100
            ):
                continue
            raise RuntimeError(
                "text preprocessing must use full-sequence labels"
            )
        for sequence_end in sequence_ends:
            if labels[sequence_end] != policy.eos_token_id:
                raise RuntimeError(
                    "text preprocessing EOS tokens must be supervised"
                )


def format_alpaca_examples(
    examples: Mapping[str, Sequence[str]],
    *,
    end_of_sequence: str,
) -> dict[str, list[str]]:
    texts = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(
            ALPACA_PROMPT.format(instruction, input_text, output_text) + end_of_sequence
        )

    return {"text": texts}


def validate_gguf_result(export_result: Mapping[str, Any]) -> Path:
    gguf_files = export_result.get("gguf_files", [])
    if len(gguf_files) != 1:
        raise RuntimeError(f"expected exactly one GGUF output, found {gguf_files}")

    return Path(gguf_files[0])


def staged_gguf_path(gguf_file: Path | str, artifact_directory: Path | str) -> Path:
    return Path(artifact_directory) / Path(gguf_file).name


def stage_gguf_artifact(
    gguf_file: Path | str,
    artifact_directory: Path | str = ARTIFACT_DIRECTORY,
) -> Path:
    source_path = Path(gguf_file)
    destination_path = staged_gguf_path(source_path, artifact_directory)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def cleanup_gguf_export(export_directory: Path | str) -> None:
    export_path = Path(export_directory)
    shutil.rmtree(export_path, ignore_errors=True)
    shutil.rmtree(Path(f"{export_path}_gguf"), ignore_errors=True)


def require_dpo_tokenizer(processing_class: Any) -> None:
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError("DPO training requires a tokenizer EOS token")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise RuntimeError("DPO training requires a tokenizer EOS token ID")


def dpo_prepared_token_ids(
    record: Mapping[str, Any],
    *,
    field: str,
    record_index: int,
) -> list[int]:
    if field not in record:
        raise RuntimeError(
            f'DPO trainer prepared preference record {record_index} without "{field}"'
        )

    raw_token_ids = sequence_values(
        record[field],
        description=(
            f'preference record {record_index} field "{field}" token IDs'
        ),
        preprocessing="DPO",
    )
    token_ids = []
    for token_index, raw_token_id in enumerate(raw_token_ids):
        try:
            token_id = operator.index(raw_token_id)
        except TypeError as error:
            raise RuntimeError(
                "DPO preprocessing preference record "
                f'{record_index} field "{field}" token ID {token_index} '
                "must be an integer"
            ) from error
        if isinstance(raw_token_id, bool) or token_id < 0:
            raise RuntimeError(
                "DPO preprocessing preference record "
                f'{record_index} field "{field}" token ID {token_index} '
                "must be a non-negative integer"
            )
        token_ids.append(token_id)

    return token_ids


def effective_dpo_completion_token_ids(
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    *,
    max_length: int,
    is_encoder_decoder: bool,
) -> list[int]:
    if is_encoder_decoder:
        return list(completion_token_ids)

    # Pinned TRL right-flushes prompt + completion, keeps the final max_length
    # positions, and applies completion loss only where the completion mask is 1.
    first_retained_index = max(
        0,
        len(prompt_token_ids) + len(completion_token_ids) - max_length,
    )
    completion_offset = max(
        0,
        first_retained_index - len(prompt_token_ids),
    )
    return list(completion_token_ids[completion_offset:])


def validate_prepared_dpo_dataset(
    train_dataset: Any,
    *,
    max_length: int,
    is_encoder_decoder: bool,
) -> None:
    for record_index, record in enumerate(train_dataset):
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"DPO trainer prepared preference record {record_index} "
                "as a non-mapping value"
            )

        prompt_token_ids = dpo_prepared_token_ids(
            record,
            field="prompt_input_ids",
            record_index=record_index,
        )
        chosen_token_ids = dpo_prepared_token_ids(
            record,
            field="chosen_input_ids",
            record_index=record_index,
        )
        rejected_token_ids = dpo_prepared_token_ids(
            record,
            field="rejected_input_ids",
            record_index=record_index,
        )
        if not chosen_token_ids or not rejected_token_ids:
            raise RuntimeError(
                f"DPO trainer prepared preference record {record_index} "
                "with an empty completion token sequence"
            )

        effective_chosen_token_ids = effective_dpo_completion_token_ids(
            prompt_token_ids,
            chosen_token_ids,
            max_length=max_length,
            is_encoder_decoder=is_encoder_decoder,
        )
        effective_rejected_token_ids = effective_dpo_completion_token_ids(
            prompt_token_ids,
            rejected_token_ids,
            max_length=max_length,
            is_encoder_decoder=is_encoder_decoder,
        )
        if effective_chosen_token_ids == effective_rejected_token_ids:
            raise RuntimeError(
                f"DPO trainer prepared preference record {record_index} with "
                "token-identical chosen and rejected completions after "
                "tokenization and effective truncation"
            )


def validate_dpo_trainer_contract(
    trainer: Any,
    *,
    policy_model: Any,
    objective: TrainingObjectiveSpec,
    max_seq_length: int,
) -> None:
    if getattr(trainer, "ref_model", object()) is not None:
        raise RuntimeError(
            "DPO trainer must use ref_model=None so the disabled policy "
            "adapter supplies reference behavior"
        )
    if getattr(trainer, "is_peft_model", False) is not True:
        raise RuntimeError("DPO trainer did not recognize the policy as a PEFT model")
    if getattr(trainer, "reference_free", True) is not False:
        raise RuntimeError("DPO trainer must not use reference-free training")
    if getattr(trainer, "model", None) is not policy_model:
        raise RuntimeError(
            "DPO trainer replaced the policy model that AIKit saves after training"
        )

    accelerator = getattr(trainer, "accelerator", None)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if not callable(unwrap_model):
        raise RuntimeError("DPO trainer does not expose model unwrapping")
    try:
        effective_model = unwrap_model(trainer.model)
    except Exception:
        raise RuntimeError("DPO trainer policy model could not be unwrapped") from None
    if not callable(getattr(effective_model, "disable_adapter", None)):
        raise RuntimeError(
            "DPO trainer policy model does not support disabling its PEFT adapter"
        )

    if getattr(trainer, "beta", None) != objective.beta:
        raise RuntimeError("DPO trainer beta does not match the configured objective")
    effective_loss_types = getattr(trainer, "loss_type", None)
    if isinstance(effective_loss_types, str):
        effective_loss_types = [effective_loss_types]
    if effective_loss_types != [objective.loss_type]:
        raise RuntimeError(
            "DPO trainer loss type does not match the configured objective"
        )
    if getattr(trainer, "max_prompt_length", None) != objective.max_prompt_length:
        raise RuntimeError(
            "DPO trainer max prompt length does not match the configured objective"
        )
    if getattr(trainer, "max_length", None) != max_seq_length:
        raise RuntimeError(
            "DPO trainer max length does not match config.unsloth.maxSeqLength"
        )
    if getattr(trainer, "max_completion_length", object()) is not None:
        raise RuntimeError(
            "DPO trainer must not apply a separate completion truncation limit"
        )
    if (
        getattr(trainer, "truncation_mode", None)
        != DPO_TRUNCATION_KEEP_END
    ):
        raise RuntimeError(
            "DPO trainer must use keep_end full-sequence truncation"
        )

    train_dataset = getattr(trainer, "train_dataset", None)
    try:
        record_count = len(train_dataset)
    except (TypeError, AttributeError):
        raise RuntimeError("DPO trainer does not expose a sized training dataset") from None
    if record_count == 0:
        raise RuntimeError("DPO trainer prepared an empty training dataset")

    validate_prepared_dpo_dataset(
        train_dataset,
        max_length=max_seq_length,
        is_encoder_decoder=bool(
            getattr(trainer, "is_encoder_decoder", False)
        ),
    )


def load_train_dependencies() -> TrainDependencies:
    # Unsloth must be imported before Transformers-based training dependencies.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from unsloth.models.loader_utils import get_model_name
    from unsloth_zoo.dataset_utils import get_chat_template_parts
    from datasets import Dataset, concatenate_datasets, load_dataset
    from huggingface_hub import model_info
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    return TrainDependencies(
        fast_language_model=FastLanguageModel,
        is_bfloat16_supported=is_bfloat16_supported,
        dataset_from_dict=Dataset.from_dict,
        load_dataset=load_dataset,
        model_info=model_info,
        resolve_model_name=get_model_name,
        sft_config=SFTConfig,
        sft_trainer=SFTTrainer,
        dpo_config=DPOConfig,
        dpo_trainer=DPOTrainer,
        get_chat_template_parts=get_chat_template_parts,
        train_on_responses_only=train_on_responses_only,
        concatenate_datasets=concatenate_datasets,
    )


def load_export_dependencies() -> ExportDependencies:
    from huggingface_hub import snapshot_download
    from unsloth import FastLanguageModel

    return ExportDependencies(
        fast_language_model=FastLanguageModel,
        snapshot_download=snapshot_download,
    )


def save_trained_model(
    model: Any,
    tokenizer: Any,
    trained_model_directory: Path | str,
) -> Path:
    trained_model_path = Path(trained_model_directory)
    trained_model_path.mkdir(parents=True, exist_ok=True)
    validate_adapter_save_contract(model)
    model.save_pretrained(
        trained_model_path,
        safe_serialization=True,
        selected_adapters=[DEFAULT_ADAPTER_NAME],
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(trained_model_path)
    validate_portable_adapter_bundle(trained_model_path)
    return trained_model_path


def train_model(
    train_config: Mapping[str, Any],
    *,
    trained_model_directory: Path | str = TRAINED_MODEL_DIRECTORY,
    dependencies: TrainDependencies | None = None,
) -> Path:
    objective = training_objective_spec(train_config)
    dataset_specs = training_dataset_specs(train_config)

    if objective.objective_type == OBJECTIVE_TYPE_DPO:
        if len(dataset_specs) != 1:
            raise ValueError(
                "DPO objective requires exactly one preference dataset"
            )
        loss = validate_training_objective(
            train_config,
            objective=objective,
            dataset_spec=dataset_specs[0],
        )
        compatibility = None
    else:
        loss = configured_training_loss(train_config)
        compatibility = training_dataset_compatibility(
            dataset_specs,
            loss=loss,
        )
        validate_response_packing(train_config, loss=loss)

    if dependencies is None:
        dependencies = load_train_dependencies()

    cfg = unsloth_config(train_config)
    max_seq_length = cfg["maxSeqLength"]
    learning_rate = normalize_go_yaml_float(
        cfg["learningRate"],
        description="config.unsloth.learningRate",
        allow_zero=False,
    )
    weight_decay = normalize_go_yaml_float(
        cfg["weightDecay"],
        description="config.unsloth.weightDecay",
        allow_zero=True,
    )

    source_datasets = []
    for dataset_spec in dataset_specs:
        dataset_index = dataset_spec.index
        if dataset_index is None:
            raise RuntimeError("parsed training dataset is missing its index")
        with indexed_dataset_errors(dataset_index):
            source_dataset = load_training_dataset(
                dataset_spec,
                load_dataset=dependencies.load_dataset,
            )
            validation_source = None
            if (
                dataset_spec.dataset_type in CHAT_DATASET_TYPES
                or dataset_spec.dataset_type == DATASET_TYPE_PREFERENCE
                or dataset_spec.dataset_type == DATASET_TYPE_TEXT
            ):
                validation_source = dataset_spec.source
            source_dataset = project_training_dataset(
                source_dataset,
                dataset_type=dataset_spec.dataset_type,
                source=validation_source,
            )
            validate_training_dataset(
                source_dataset,
                dataset_type=dataset_spec.dataset_type,
                source=validation_source,
            )
            if dataset_spec.dataset_type == DATASET_TYPE_SHAREGPT:
                source_dataset = normalize_sharegpt_dataset(
                    source_dataset,
                    source=dataset_spec.source,
                )
            if (
                objective.objective_type == OBJECTIVE_TYPE_SFT
                and dataset_spec.dataset_type in CHAT_DATASET_TYPES
                and loss == LOSS_RESPONSE
            ):
                validate_response_training_dataset(
                    source_dataset,
                    dataset_type=dataset_spec.dataset_type,
                    source=dataset_spec.source,
                )
            source_datasets.append(source_dataset)

    training_model_name, training_model_revision = resolve_model_snapshot(
        train_config["baseModel"],
        load_in_4bit=cfg["loadIn4bit"],
        description="resolved training model revision",
        model_info=dependencies.model_info,
        resolve_model_name=dependencies.resolve_model_name,
    )
    model, tokenizer = dependencies.fast_language_model.from_pretrained(
        model_name=training_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg["loadIn4bit"],
        revision=training_model_revision,
        use_exact_model_name=True,
    )

    dataset = None
    has_chat_dataset = False
    has_text_dataset = False
    response_only = False
    completion_only = False
    validate_all_full_sequence_records = False
    response_markers = None
    text_policy = None
    prompt_prefix_fingerprint = None
    messages_token_fingerprint = None

    if objective.objective_type == OBJECTIVE_TYPE_DPO:
        require_dpo_tokenizer(tokenizer)
        dataset = source_datasets[0]
    else:
        has_chat_dataset = any(
            dataset_spec.dataset_type in CHAT_DATASET_TYPES
            for dataset_spec in dataset_specs
        )
        has_text_dataset = any(
            dataset_spec.dataset_type == DATASET_TYPE_TEXT
            for dataset_spec in dataset_specs
        )
        response_only = compatibility == DATASET_COMPATIBILITY_RESPONSE_CHAT
        completion_only = (
            compatibility == DATASET_COMPATIBILITY_PROMPT_COMPLETION
        )
        validate_all_full_sequence_records = (
            compatibility == DATASET_COMPATIBILITY_FULL_SEQUENCE
            and (len(dataset_specs) > 1 or has_chat_dataset)
        )

        if has_chat_dataset:
            require_messages_chat_template(tokenizer)
            if response_only:
                response_markers = derive_response_markers(
                    tokenizer,
                    get_chat_template_parts=(
                        dependencies.get_chat_template_parts
                    ),
                )

        if has_text_dataset:
            text_policy = text_boundary_policy(tokenizer)

        prompt_add_special_tokens = None
        if completion_only:
            first_prompt_record = next(iter(source_datasets[0]))
            prompt_add_special_tokens = prompt_completion_add_special_tokens(
                tokenizer,
                first_prompt_record["prompt"],
            )

        rendered_source_datasets = []
        canonical_chat_fingerprints = []
        prompt_prefix_fingerprints = []
        for dataset_spec, source_dataset in zip(
            dataset_specs,
            source_datasets,
        ):
            dataset_index = dataset_spec.index
            if dataset_index is None:
                raise RuntimeError(
                    "parsed training dataset is missing its index"
                )
            with indexed_dataset_errors(dataset_index):
                if dataset_spec.dataset_type in CHAT_DATASET_TYPES:
                    messages_source_dataset = source_dataset
                    source_dataset = render_messages_dataset(
                        messages_source_dataset,
                        processing_class=tokenizer,
                        source=dataset_spec.source,
                        dataset_type=dataset_spec.dataset_type,
                    )
                    canonical_chat_fingerprints.append(
                        validate_messages_tokenization(
                            messages_source_dataset,
                            source_dataset,
                            processing_class=tokenizer,
                            max_seq_length=max_seq_length,
                            source=dataset_spec.source,
                            dataset_type=dataset_spec.dataset_type,
                            response_markers=response_markers,
                        )
                    )
                elif (
                    dataset_spec.dataset_type
                    == DATASET_TYPE_PROMPT_COMPLETION
                ):
                    prompt_prefix_fingerprints.append(
                        validate_prompt_completion_tokenization(
                            source_dataset,
                            processing_class=tokenizer,
                            max_seq_length=max_seq_length,
                            add_special_tokens=prompt_add_special_tokens,
                        )
                    )
                elif dataset_spec.dataset_type == DATASET_TYPE_TEXT:
                    if text_policy is None:
                        raise RuntimeError(
                            "text preprocessing did not produce a boundary "
                            "policy"
                        )
                    validate_text_sequence_lengths(
                        source_dataset,
                        processing_class=tokenizer,
                        policy=text_policy,
                        max_seq_length=max_seq_length,
                        source=dataset_spec.source,
                    )
                rendered_source_datasets.append(source_dataset)

    base_model_name, base_model_revision = resolve_export_base_model(
        train_config["baseModel"],
        model_info=dependencies.model_info,
        resolve_model_name=dependencies.resolve_model_name,
    )

    model = dependencies.fast_language_model.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        use_rslora=False,
        loftq_config=None,
        base_model_name_or_path=base_model_name,
        revision=base_model_revision,
    )
    pin_peft_base_model(
        model,
        base_model_name=base_model_name,
        revision=base_model_revision,
    )

    if objective.objective_type == OBJECTIVE_TYPE_DPO:
        if not callable(getattr(model, "disable_adapter", None)):
            raise RuntimeError(
                "DPO policy model does not support disabling its PEFT adapter"
            )
        bfloat16_supported = dependencies.is_bfloat16_supported()
        trainer = dependencies.dpo_trainer(
            model=model,
            ref_model=None,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=dependencies.dpo_config(
                output_dir="outputs",
                dataset_num_proc=2,
                max_length=max_seq_length,
                max_prompt_length=objective.max_prompt_length,
                max_completion_length=None,
                truncation_mode=DPO_TRUNCATION_KEEP_END,
                beta=objective.beta,
                loss_type=objective.loss_type,
                reference_free=False,
                per_device_train_batch_size=cfg["batchSize"],
                gradient_accumulation_steps=cfg[
                    "gradientAccumulationSteps"
                ],
                warmup_steps=cfg["warmupSteps"],
                max_steps=cfg["maxSteps"],
                learning_rate=learning_rate,
                fp16=not bfloat16_supported,
                bf16=bfloat16_supported,
                logging_steps=cfg["loggingSteps"],
                optim=cfg["optimizer"],
                weight_decay=weight_decay,
                lr_scheduler_type=cfg["lrSchedulerType"],
                seed=cfg["seed"],
                save_strategy="no",
                report_to="none",
            ),
        )
        validate_dpo_trainer_contract(
            trainer,
            policy_model=model,
            objective=objective,
            max_seq_length=max_seq_length,
        )
        trainer.train()
        return save_trained_model(
            model,
            tokenizer,
            trained_model_directory,
        )

    canonical_datasets = []
    for dataset_spec, source_dataset in zip(
        dataset_specs,
        rendered_source_datasets,
    ):
        dataset_index = dataset_spec.index
        if dataset_index is None:
            raise RuntimeError("parsed training dataset is missing its index")
        with indexed_dataset_errors(dataset_index):
            source_dataset = prepare_training_dataset(
                source_dataset,
                dataset_type=dataset_spec.dataset_type,
                end_of_sequence=tokenizer.eos_token,
                text_policy=text_policy,
            )

            if len(dataset_specs) > 1:
                canonical_fields = (
                    DATASET_REQUIRED_FIELDS[DATASET_TYPE_PROMPT_COMPLETION]
                    if completion_only
                    else DATASET_REQUIRED_FIELDS[DATASET_TYPE_TEXT]
                )
                source_dataset = normalize_canonical_string_dataset(
                    source_dataset,
                    fields=canonical_fields,
                    dataset_type=dataset_spec.dataset_type,
                    source=dataset_spec.source,
                    dataset_index=dataset_index,
                )

            canonical_datasets.append(source_dataset)

    messages_token_fingerprints = []
    if response_only or validate_all_full_sequence_records:
        if len(canonical_datasets) == 1 and has_chat_dataset:
            messages_token_fingerprints = canonical_chat_fingerprints
        else:
            first_record = next(iter(canonical_datasets[0]))
            full_sequence_add_special_tokens = (
                messages_unsloth_add_special_tokens(
                    tokenizer,
                    first_record["text"],
                )
            )
            for dataset_spec, canonical_dataset in zip(
                dataset_specs,
                canonical_datasets,
            ):
                dataset_index = dataset_spec.index
                if dataset_index is None:
                    raise RuntimeError(
                        "parsed training dataset is missing its index"
                    )
                with indexed_dataset_errors(dataset_index):
                    messages_token_fingerprints.append(
                        validate_full_sequence_text_tokenization(
                            canonical_dataset,
                            processing_class=tokenizer,
                            max_seq_length=max_seq_length,
                            dataset_type=dataset_spec.dataset_type,
                            source=dataset_spec.source,
                            dataset_index=dataset_index,
                            add_special_tokens=(
                                full_sequence_add_special_tokens
                            ),
                        )
                    )

    if len(canonical_datasets) == 1:
        dataset = canonical_datasets[0]
    else:
        if not callable(dependencies.concatenate_datasets):
            raise RuntimeError(
                "multiple training datasets require "
                "datasets.concatenate_datasets"
            )
        dataset = dependencies.concatenate_datasets(canonical_datasets)

    if response_only or validate_all_full_sequence_records:
        messages_token_fingerprint = merge_messages_token_fingerprints(
            messages_token_fingerprints
        )
    if completion_only:
        prompt_prefix_fingerprint = merge_prompt_prefix_fingerprints(
            prompt_prefix_fingerprints
        )

    bfloat16_supported = dependencies.is_bfloat16_supported()

    trainer = dependencies.sft_trainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=dependencies.sft_config(
            output_dir="outputs",
            dataset_text_field="text",
            dataset_num_proc=2,
            assistant_only_loss=False,
            completion_only_loss=completion_only,
            max_length=max_seq_length,
            packing=cfg["packing"],
            per_device_train_batch_size=cfg["batchSize"],
            gradient_accumulation_steps=cfg["gradientAccumulationSteps"],
            warmup_steps=cfg["warmupSteps"],
            max_steps=cfg["maxSteps"],
            learning_rate=learning_rate,
            fp16=not bfloat16_supported,
            bf16=bfloat16_supported,
            logging_steps=cfg["loggingSteps"],
            optim=cfg["optimizer"],
            weight_decay=weight_decay,
            lr_scheduler_type=cfg["lrSchedulerType"],
            seed=cfg["seed"],
            save_strategy="no",
            report_to="none",
        ),
    )
    if response_only:
        if bool(getattr(trainer.args, "packing", False)):
            raise RuntimeError(
                "response-only messages preprocessing does not support "
                "effective trainer packing because response masks must not "
                "cross conversation boundaries"
            )
        if response_markers is None:
            raise RuntimeError(
                "response-only messages preprocessing did not derive markers"
            )
        marker_kwargs = {"force_match": True}
        if not response_markers.use_tokenizer_parts:
            marker_kwargs.update(
                instruction_part=response_markers.instruction_part,
                response_part=response_markers.response_part,
            )
        trainer = dependencies.train_on_responses_only(
            trainer,
            **marker_kwargs,
        )
        if trainer is None:
            raise RuntimeError(
                "response-only messages preprocessing did not return a trainer"
            )
    if completion_only:
        # This is exercised by the GPU smoke path against the exact locked,
        # Unsloth-patched TRL trainer before any training step can silently use
        # full-sequence loss or omit EOS supervision.
        verify_prompt_completion_preprocessing(
            trainer,
            dataset_from_dict=dependencies.dataset_from_dict,
            processing_class=tokenizer,
        )
        validate_prepared_prompt_completion_dataset(
            trainer.train_dataset,
            eos_token_id=tokenizer.eos_token_id,
            max_seq_length=max_seq_length,
            packing=bool(getattr(trainer.args, "packing", False)),
            packing_strategy=getattr(
                trainer.args,
                "packing_strategy",
                "bfd",
            ),
            expected_prompt_prefix_fingerprint=prompt_prefix_fingerprint,
        )

    if response_only or validate_all_full_sequence_records:
        if messages_token_fingerprint is None:
            raise RuntimeError(
                "messages preprocessing did not produce a source fingerprint"
            )
        validate_prepared_messages_dataset(
            trainer.train_dataset,
            data_collator=trainer.data_collator,
            max_seq_length=max_seq_length,
            packing=bool(getattr(trainer.args, "packing", False)),
            padding_free=bool(getattr(trainer.args, "padding_free", False)),
            packing_strategy=getattr(
                trainer.args,
                "packing_strategy",
                "bfd",
            ),
            expected_fingerprint=messages_token_fingerprint,
            loss=loss,
            response_markers=response_markers,
        )

    if has_text_dataset:
        if text_policy is None:
            raise RuntimeError(
                "text preprocessing did not produce a boundary policy"
            )
        verify_text_preprocessing(
            trainer,
            dataset_from_dict=dependencies.dataset_from_dict,
            processing_class=tokenizer,
            policy=text_policy,
        )
    trainer.train()
    return save_trained_model(model, tokenizer, trained_model_directory)

def export_model(
    export_config: Mapping[str, Any],
    *,
    trained_model_directory: Path | str = TRAINED_MODEL_DIRECTORY,
    export_directory: Path | str = EXPORT_DIRECTORY,
    artifact_directory: Path | str = ARTIFACT_DIRECTORY,
    dependencies: ExportDependencies | None = None,
) -> Path:
    if dependencies is None:
        dependencies = load_export_dependencies()

    cfg = unsloth_config(export_config)
    trained_model_path = Path(trained_model_directory)
    export_path = Path(export_directory)

    validate_portable_adapter_bundle(trained_model_path)
    with tempfile.TemporaryDirectory(prefix="aikit-gguf-adapter-") as temp_dir:
        export_adapter_path = Path(temp_dir) / "adapter"
        shutil.copytree(trained_model_path, export_adapter_path)
        pin_adapter_base_model_snapshot(
            export_adapter_path,
            snapshot_download=dependencies.snapshot_download,
        )

        model, tokenizer = dependencies.fast_language_model.from_pretrained(
            model_name=str(export_adapter_path),
            max_seq_length=cfg["maxSeqLength"],
            dtype=None,
            load_in_4bit=cfg["loadIn4bit"],
            local_files_only=True,
        )
        export_result = model.save_pretrained_gguf(
            export_path,
            tokenizer,
            quantization_method=output_config(export_config)["quantize"],
        )
    gguf_file = validate_gguf_result(export_result)
    staged_file = stage_gguf_artifact(gguf_file, artifact_directory)
    cleanup_gguf_export(export_path)
    return staged_file


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AIKit Unsloth phase.")
    parser.add_argument("mode", choices=("train", "export"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)

    if args.mode == "train":
        train_config = load_config(TRAIN_CONFIG_PATH)
        print("Loaded fine-tuning configuration.")
        train_model(train_config)
        return

    export_config = load_config(EXPORT_CONFIG_PATH)
    print("Loaded export configuration.")
    export_model(export_config)


if __name__ == "__main__":
    main()
