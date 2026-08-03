#!/usr/bin/env python3

import argparse
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

# Alpaca is the only dataset type currently supported by the AIKit fine-tuning API.
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


class TrainDependencies(NamedTuple):
    fast_language_model: Any
    is_bfloat16_supported: Callable[[], bool]
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
    model: Any,
    configured_model_name: str,
    *,
    model_info: Callable[..., Any],
    resolve_model_name: Callable[..., str],
) -> tuple[str, str]:
    base_model_name = resolve_model_name(
        configured_model_name,
        load_in_4bit=False,
    )
    if base_model_name == configured_model_name:
        config = getattr(model, "config", None)
        revision = getattr(config, "_commit_hash", None)
    else:
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
    from datasets import load_dataset
    from huggingface_hub import model_info
    from trl import SFTConfig, SFTTrainer

    return TrainDependencies(
        fast_language_model=FastLanguageModel,
        is_bfloat16_supported=is_bfloat16_supported,
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
    if dependencies is None:
        dependencies = load_train_dependencies()

    cfg = unsloth_config(train_config)
    max_seq_length = cfg["maxSeqLength"]

    model, tokenizer = dependencies.fast_language_model.from_pretrained(
        model_name=train_config["baseModel"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg["loadIn4bit"],
    )
    base_model_name, base_model_revision = resolve_export_base_model(
        model,
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

    source = train_config["datasets"][0]["source"]
    load_spec = dataset_load_spec(source)
    dataset = dependencies.load_dataset(load_spec.path, **load_spec.kwargs)
    dataset = dataset.map(
        partial(format_alpaca_examples, end_of_sequence=tokenizer.eos_token),
        batched=True,
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
