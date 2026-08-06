#!/usr/bin/env python3

import argparse
import copy
import json
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
DATASET_TYPE_PROMPT_COMPLETION = "prompt-completion"
SUPPORTED_DATASET_TYPES = frozenset(
    (DATASET_TYPE_ALPACA, DATASET_TYPE_PROMPT_COMPLETION)
)
DATASET_REQUIRED_FIELDS = {
    DATASET_TYPE_ALPACA: ("instruction", "input", "output"),
    DATASET_TYPE_PROMPT_COMPLETION: ("prompt", "completion"),
}
PREPROCESSING_VERIFICATION_PROMPT = "Question: What is 2 + 2?\nAnswer:"
PREPROCESSING_VERIFICATION_COMPLETION = " 4."

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


def validate_training_dataset(dataset: Any, *, dataset_type: str) -> None:
    required_fields = DATASET_REQUIRED_FIELDS[dataset_type]
    record_count = 0

    for record_index, record in enumerate(dataset):
        record_count += 1
        if not isinstance(record, Mapping):
            raise ValueError(
                f"{dataset_type} dataset record {record_index} must be a mapping"
            )

        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f'{dataset_type} dataset record {record_index} is missing required field "{field}"'
                )

            value = record[field]
            if not isinstance(value, str):
                raise ValueError(
                    f'{dataset_type} dataset record {record_index} field "{field}" must be a string'
                )

        if (
            dataset_type == DATASET_TYPE_PROMPT_COMPLETION
            and record["completion"] == ""
        ):
            raise ValueError(
                f'{dataset_type} dataset record {record_index} field "completion" must be a non-empty string'
            )

    if record_count == 0:
        raise ValueError(f"{dataset_type} dataset must contain at least one record")


def project_training_dataset(dataset: Any, *, dataset_type: str) -> Any:
    if len(dataset) == 0:
        raise ValueError(f"{dataset_type} dataset must contain at least one record")

    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, Sequence) or isinstance(
        column_names, (str, bytes)
    ):
        raise ValueError(f"{dataset_type} dataset does not expose its columns")

    required_fields = DATASET_REQUIRED_FIELDS[dataset_type]
    missing_fields = [
        field for field in required_fields if field not in column_names
    ]
    if missing_fields:
        quoted_fields = ", ".join(f'"{field}"' for field in missing_fields)
        raise ValueError(
            f"{dataset_type} dataset is missing required columns: {quoted_fields}"
        )

    return dataset.select_columns(list(required_fields))


def prepare_training_dataset(
    dataset: Any,
    *,
    dataset_type: str,
    end_of_sequence: str,
) -> Any:
    if dataset_type == DATASET_TYPE_ALPACA:
        return dataset.map(
            partial(format_alpaca_examples, end_of_sequence=end_of_sequence),
            batched=True,
        )
    if dataset_type == DATASET_TYPE_PROMPT_COMPLETION:
        return dataset

    raise ValueError(f"unsupported dataset type {dataset_type!r}")


def sequence_values(value: Any, *, description: str) -> list[Any]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(
            f"prompt-completion preprocessing {description} must be a sequence"
        )

    return list(value)


def single_batch_row(value: Any, *, description: str) -> list[Any]:
    rows = sequence_values(value, description=description)
    if len(rows) != 1:
        raise RuntimeError(
            f"prompt-completion preprocessing {description} must contain one row"
        )

    return sequence_values(rows[0], description=f"{description} row")


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

    bos_token = getattr(processing_class, "bos_token", None)
    chat_template = getattr(processing_class, "chat_template", "") or ""
    add_special_tokens = not (
        bos_token is not None
        and (
            PREPROCESSING_VERIFICATION_PROMPT.startswith(bos_token)
            or bos_token in chat_template
        )
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
    dataset = dependencies.load_dataset(load_spec.path, **load_spec.kwargs)
    dataset = project_training_dataset(
        dataset,
        dataset_type=dataset_spec.dataset_type,
    )
    validate_training_dataset(dataset, dataset_type=dataset_spec.dataset_type)

    model, tokenizer = dependencies.fast_language_model.from_pretrained(
        model_name=train_config["baseModel"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg["loadIn4bit"],
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
