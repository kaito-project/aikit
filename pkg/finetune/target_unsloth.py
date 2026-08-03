#!/usr/bin/env python3

import shutil
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import yaml


with open("/config.yaml", "r", encoding="utf-8") as config_file:
    data = yaml.safe_load(config_file)
print("Loaded fine-tuning configuration.")

cfg = data["config"]["unsloth"]
max_seq_length = cfg["maxSeqLength"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=data["baseModel"],
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=cfg["loadIn4bit"],
)

model = FastLanguageModel.get_peft_model(
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
)

# Alpaca is the only dataset type currently supported by the AIKit fine-tuning API.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

eos_token = tokenizer.eos_token


def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(alpaca_prompt.format(instruction, input_text, output_text) + eos_token)
    return {"text": texts}


source = data["datasets"][0]["source"]
if source.startswith("http"):
    dataset = load_dataset("json", data_files={"train": source}, split="train")
else:
    dataset = load_dataset(source, split="train")

dataset = dataset.map(formatting_prompts_func, batched=True)
bfloat16_supported = is_bfloat16_supported()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=SFTConfig(
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

export_directory = Path("/aikit-unsloth-export")
export_result = model.save_pretrained_gguf(
    export_directory,
    tokenizer,
    quantization_method=data["output"]["quantize"],
)
gguf_files = export_result.get("gguf_files", [])
if len(gguf_files) != 1:
    raise RuntimeError(f"expected exactly one GGUF output, found {gguf_files}")

artifact_directory = Path("/model")
artifact_directory.mkdir(parents=True, exist_ok=True)
Path(gguf_files[0]).replace(artifact_directory / Path(gguf_files[0]).name)
shutil.rmtree(export_directory, ignore_errors=True)
shutil.rmtree(Path(f"{export_directory}_gguf"), ignore_errors=True)
