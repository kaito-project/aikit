#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import operator
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

TRAIN_CONFIG_PATH = Path("/aikit-config/train-config.yaml")
EXPORT_CONFIG_PATH = Path("/aikit-config/export-config.yaml")
TRAINED_MODEL_DIRECTORY = Path("/aikit-trained-model")
EXPORT_DIRECTORY = Path("/aikit-unsloth-export")
ARTIFACT_DIRECTORY = Path("/model")
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
HF_COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATASET_TYPE_ALPACA = "alpaca"
DATASET_TYPE_MESSAGES = "messages"
DATASET_TYPE_PROMPT_COMPLETION = "prompt-completion"
DATASET_TYPE_TEXT = "text"
SUPPORTED_DATASET_TYPES = frozenset(
    (
        DATASET_TYPE_ALPACA,
        DATASET_TYPE_MESSAGES,
        DATASET_TYPE_PROMPT_COMPLETION,
        DATASET_TYPE_TEXT,
    )
)
DATASET_REQUIRED_FIELDS = {
    DATASET_TYPE_ALPACA: ("instruction", "input", "output"),
    DATASET_TYPE_MESSAGES: ("messages",),
    DATASET_TYPE_PROMPT_COMPLETION: ("prompt", "completion"),
    DATASET_TYPE_TEXT: ("text",),
}
MESSAGE_FIELDS = frozenset(("role", "content"))
SUPPORTED_MESSAGE_ROLES = frozenset(("system", "user", "assistant"))
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


class TrainingDatasetSpec(NamedTuple):
    source: str
    dataset_type: str


class PromptPrefixFingerprint(NamedTuple):
    sequence_count: int
    first_digest_sum: int
    second_digest_sum: int


class MessagesTokenFingerprint(NamedTuple):
    sequence_count: int
    first_digest_sum: int
    second_digest_sum: int


class MessagesRenderError(RuntimeError):
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


def require_hf_commit_hash(revision: Any, *, description: str) -> str:
    if (
        not isinstance(revision, str)
        or HF_COMMIT_HASH_PATTERN.fullmatch(revision) is None
    ):
        raise RuntimeError(
            f"{description} is not an immutable Hugging Face commit hash"
        )

    return revision


def resolve_export_base_model(
    configured_model_name: str,
    *,
    model_info: Callable[..., Any],
    resolve_model_name: Callable[..., str],
) -> tuple[str, str]:
    base_model_name = resolve_model_name(
        configured_model_name,
        load_in_4bit=False,
    )
    revision = getattr(model_info(repo_id=base_model_name), "sha", None)

    return base_model_name, require_hf_commit_hash(
        revision,
        description="resolved export base model revision",
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


def classify_dataset_source(source: str) -> DatasetSourceKind:
    parsed_source = urlparse(source)
    if parsed_source.scheme.lower() in {"http", "https"} and parsed_source.netloc:
        return DatasetSourceKind.JSON_URL

    return DatasetSourceKind.DATASET


def dataset_load_spec(source: str) -> DatasetLoadSpec:
    if classify_dataset_source(source) is DatasetSourceKind.JSON_URL:
        return DatasetLoadSpec(
            path="json",
            kwargs={"data_files": {"train": source}, "split": "train"},
        )

    return DatasetLoadSpec(path=source, kwargs={"split": "train"})


def dataset_source_description(source: str) -> str:
    if classify_dataset_source(source) is DatasetSourceKind.JSON_URL:
        return "remote JSON URL"

    return f"source {source!r}"


def training_dataset_spec(
    train_config: Mapping[str, Any],
) -> TrainingDatasetSpec:
    datasets = train_config.get("datasets")
    if (
        not isinstance(datasets, Sequence)
        or isinstance(datasets, (str, bytes))
        or len(datasets) != 1
    ):
        raise ValueError("training configuration must define exactly one dataset")

    dataset = datasets[0]
    if not isinstance(dataset, Mapping):
        raise ValueError("training dataset configuration must be a mapping")

    dataset_type = dataset.get("type")
    if (
        not isinstance(dataset_type, str)
        or dataset_type not in SUPPORTED_DATASET_TYPES
    ):
        raise ValueError(f"unsupported dataset type {dataset_type!r}")

    source = dataset.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("training dataset source must be a non-empty string")

    return TrainingDatasetSpec(source=source, dataset_type=dataset_type)


def dataset_error_subject(
    dataset_type: str,
    *,
    source: str | None = None,
    record_index: int | None = None,
) -> str:
    if source is None:
        subject = f"{dataset_type} dataset"
        if record_index is not None:
            subject += f" record {record_index}"
        return subject

    subject = f"{dataset_type} dataset {dataset_source_description(source)}"
    if record_index is not None:
        subject += f" row {record_index}"
    return subject


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

        if dataset_type == DATASET_TYPE_MESSAGES:
            validate_messages_top_level_fields(record, subject=subject)

        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f'{subject} is missing required field "{field}"'
                )

            value = record[field]
            if dataset_type == DATASET_TYPE_MESSAGES:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f'{subject} field "{field}" must be a string'
                )

        if dataset_type == DATASET_TYPE_MESSAGES:
            validate_messages_value(record["messages"], subject=subject)
        if (
            dataset_type == DATASET_TYPE_PROMPT_COMPLETION
            and record["completion"] == ""
        ):
            raise ValueError(
                f'{dataset_type} dataset record {record_index} field "completion" must be a non-empty string'
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

    if dataset_type == DATASET_TYPE_MESSAGES:
        validate_messages_top_level_fields(column_names, subject=subject)

    return dataset.select_columns(list(required_fields))


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
) -> dict[str, str]:
    try:
        record_index = operator.index(raw_index)
    except TypeError as error:
        raise MessagesRenderError(
            "messages preprocessing row index must be an integer"
        ) from error
    subject = (
        f"{DATASET_TYPE_MESSAGES} dataset {source_description} "
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
) -> Any:
    source_description = dataset_source_description(source)
    try:
        return dataset.map(
            partial(
                render_messages_example,
                processing_class=processing_class,
                source_description=source_description,
            ),
            batched=False,
            with_indices=True,
            remove_columns=list(dataset.column_names),
            writer_batch_size=1,
        )
    except MessagesRenderError:
        raise
    except Exception:
        subject = dataset_error_subject(DATASET_TYPE_MESSAGES, source=source)
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
    canonical_token_rows: list[list[Any]] = []

    def validate_batch() -> None:
        nonlocal fingerprint
        batch_subject = dataset_error_subject(
            DATASET_TYPE_MESSAGES,
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
        for batch_index, (canonical_ids, rendered_ids) in enumerate(
            zip(canonical_token_rows, rendered_token_rows)
        ):
            record_index = batch_start + batch_index
            subject = dataset_error_subject(
                DATASET_TYPE_MESSAGES,
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
            DATASET_TYPE_MESSAGES,
            source=source,
            record_index=record_count,
        )
        canonical_ids = messages_chat_template_token_ids(
            processing_class,
            source_record["messages"],
            max_length=token_limit,
            subject=subject,
        )
        rendered_texts.append(text)
        canonical_token_rows.append(canonical_ids)
        record_count += 1
        if len(rendered_texts) == effective_batch_size:
            validate_batch()
            rendered_texts.clear()
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
    if dataset_type == DATASET_TYPE_MESSAGES:
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


def tokenize_verification_text(
    processing_class: Any,
    text: str,
    *,
    add_special_tokens: bool,
    description: str,
) -> list[Any]:
    tokenized = processing_class(
        text,
        add_special_tokens=add_special_tokens,
    )
    if not isinstance(tokenized, Mapping):
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization must return a mapping"
        )

    try:
        return sequence_values(
            tokenized["input_ids"],
            description=f"{description} token IDs",
        )
    except KeyError as error:
        raise RuntimeError(
            f"prompt-completion preprocessing {description} tokenization did not produce input_ids"
        ) from error


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
    description: str,
) -> list[list[Any]]:
    tokenized = processing_class(
        list(texts),
        add_special_tokens=add_special_tokens,
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

    return [
        sequence_values(
            token_row,
            description=f"{description} token IDs row {row_index}",
        )
        for row_index, token_row in enumerate(token_rows)
    ]


def empty_prompt_prefix_fingerprint() -> PromptPrefixFingerprint:
    return PromptPrefixFingerprint(0, 0, 0)


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
    add_special_tokens: bool | None = None
    fingerprint = empty_prompt_prefix_fingerprint()

    def validate_batch() -> None:
        nonlocal fingerprint
        token_rows = tokenize_verification_texts(
            processing_class,
            prompts + prompt_completions,
            add_special_tokens=add_special_tokens,
            description=(
                f"records {batch_start}-{batch_start + len(prompts) - 1} "
                "prompts and prompt-completions"
            ),
        )
        prompt_token_rows = token_rows[: len(prompts)]
        prompt_completion_token_rows = token_rows[len(prompts) :]
        for batch_index, (prompt_ids, input_ids) in enumerate(
            zip(prompt_token_rows, prompt_completion_token_rows)
        ):
            record_index = batch_start + batch_index
            input_ids = input_ids[:max_seq_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
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
        if add_special_tokens is None:
            add_special_tokens = prompt_completion_add_special_tokens(
                processing_class,
                record["prompt"],
            )

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
        description="prompt",
    )
    verification_completion = PREPROCESSING_VERIFICATION_COMPLETION
    if not verification_completion.endswith(eos_token):
        verification_completion += eos_token
    expected_input_ids = tokenize_verification_text(
        processing_class,
        PREPROCESSING_VERIFICATION_PROMPT + verification_completion,
        add_special_tokens=add_special_tokens,
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
) -> None:
    """Verify actual rendered-message boundaries and full-sequence labels."""
    if packing and packing_strategy != "bfd":
        raise RuntimeError(
            "messages preprocessing validation requires the bfd packing strategy"
        )

    fingerprint = empty_messages_token_fingerprint()
    record_count = 0
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


def load_train_dependencies() -> TrainDependencies:
    # Unsloth must be imported before Transformers-based training dependencies.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.models.loader_utils import get_model_name
    from datasets import Dataset, load_dataset
    from huggingface_hub import model_info
    from trl import SFTConfig, SFTTrainer

    return TrainDependencies(
        fast_language_model=FastLanguageModel,
        is_bfloat16_supported=is_bfloat16_supported,
        dataset_from_dict=Dataset.from_dict,
        load_dataset=load_dataset,
        model_info=model_info,
        resolve_model_name=get_model_name,
        sft_config=SFTConfig,
        sft_trainer=SFTTrainer,
    )


def load_export_dependencies() -> ExportDependencies:
    from huggingface_hub import snapshot_download
    from unsloth import FastLanguageModel

    return ExportDependencies(
        fast_language_model=FastLanguageModel,
        snapshot_download=snapshot_download,
    )


def train_model(
    train_config: Mapping[str, Any],
    *,
    trained_model_directory: Path | str = TRAINED_MODEL_DIRECTORY,
    dependencies: TrainDependencies | None = None,
) -> Path:
    dataset_spec = training_dataset_spec(train_config)

    if dependencies is None:
        dependencies = load_train_dependencies()

    cfg = unsloth_config(train_config)
    max_seq_length = cfg["maxSeqLength"]

    load_spec = dataset_load_spec(dataset_spec.source)
    try:
        dataset = dependencies.load_dataset(load_spec.path, **load_spec.kwargs)
    except Exception:
        if dataset_spec.dataset_type == DATASET_TYPE_MESSAGES:
            subject = dataset_error_subject(
                DATASET_TYPE_MESSAGES,
                source=dataset_spec.source,
            )
            raise RuntimeError(f"{subject} could not be loaded") from None
        raise
    validation_source = (
        dataset_spec.source
        if dataset_spec.dataset_type
        in {DATASET_TYPE_MESSAGES, DATASET_TYPE_TEXT}
        else None
    )
    dataset = project_training_dataset(
        dataset,
        dataset_type=dataset_spec.dataset_type,
        source=validation_source,
    )
    validate_training_dataset(
        dataset,
        dataset_type=dataset_spec.dataset_type,
        source=validation_source,
    )

    model, tokenizer = dependencies.fast_language_model.from_pretrained(
        model_name=train_config["baseModel"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg["loadIn4bit"],
    )
    messages_token_fingerprint = None
    if dataset_spec.dataset_type == DATASET_TYPE_MESSAGES:
        require_messages_chat_template(tokenizer)
        messages_source_dataset = dataset
        dataset = render_messages_dataset(
            messages_source_dataset,
            processing_class=tokenizer,
            source=dataset_spec.source,
        )
        messages_token_fingerprint = validate_messages_tokenization(
            messages_source_dataset,
            dataset,
            processing_class=tokenizer,
            max_seq_length=max_seq_length,
            source=dataset_spec.source,
        )

    prompt_prefix_fingerprint = None
    if dataset_spec.dataset_type == DATASET_TYPE_PROMPT_COMPLETION:
        prompt_prefix_fingerprint = validate_prompt_completion_tokenization(
            dataset,
            processing_class=tokenizer,
            max_seq_length=max_seq_length,
        )

    text_policy = None
    if dataset_spec.dataset_type == DATASET_TYPE_TEXT:
        text_policy = text_boundary_policy(tokenizer)
        validate_text_sequence_lengths(
            dataset,
            processing_class=tokenizer,
            policy=text_policy,
            max_seq_length=max_seq_length,
            source=dataset_spec.source,
        )
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

    dataset = prepare_training_dataset(
        dataset,
        dataset_type=dataset_spec.dataset_type,
        end_of_sequence=tokenizer.eos_token,
        text_policy=text_policy,
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
            completion_only_loss=(
                dataset_spec.dataset_type == DATASET_TYPE_PROMPT_COMPLETION
            ),
            max_length=max_seq_length,
            packing=cfg["packing"],
            per_device_train_batch_size=cfg["batchSize"],
            gradient_accumulation_steps=cfg["gradientAccumulationSteps"],
            warmup_steps=cfg["warmupSteps"],
            max_steps=cfg["maxSteps"],
            learning_rate=cfg["learningRate"],
            fp16=not bfloat16_supported,
            bf16=bfloat16_supported,
            logging_steps=cfg["loggingSteps"],
            optim=cfg["optimizer"],
            weight_decay=cfg["weightDecay"],
            lr_scheduler_type=cfg["lrSchedulerType"],
            seed=cfg["seed"],
            save_strategy="no",
            report_to="none",
        ),
    )
    if dataset_spec.dataset_type == DATASET_TYPE_PROMPT_COMPLETION:
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
            packing_strategy=getattr(trainer.args, "packing_strategy", "bfd"),
            expected_prompt_prefix_fingerprint=prompt_prefix_fingerprint,
        )

    if dataset_spec.dataset_type == DATASET_TYPE_MESSAGES:
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
            packing_strategy=getattr(trainer.args, "packing_strategy", "bfd"),
            expected_fingerprint=messages_token_fingerprint,
        )

    if dataset_spec.dataset_type == DATASET_TYPE_TEXT:
        verify_text_preprocessing(
            trainer,
            dataset_from_dict=dependencies.dataset_from_dict,
            processing_class=tokenizer,
            policy=text_policy,
        )
    trainer.train()

    trained_model_path = Path(trained_model_directory)
    trained_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(trained_model_path)
    tokenizer.save_pretrained(trained_model_path)
    return trained_model_path


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

    pin_adapter_base_model_snapshot(
        trained_model_path,
        snapshot_download=dependencies.snapshot_download,
    )

    model, tokenizer = dependencies.fast_language_model.from_pretrained(
        model_name=str(trained_model_path),
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
