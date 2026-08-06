import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).with_name("target_unsloth.py")
MODULE_SPEC = importlib.util.spec_from_file_location("target_unsloth", MODULE_PATH)
target_unsloth = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(target_unsloth)


def example_train_config():
    return {
        "baseModel": "example/model",
        "datasets": [{"source": "organization/dataset", "type": "alpaca"}],
        "config": {
            "unsloth": {
                "packing": False,
                "maxSeqLength": 2048,
                "loadIn4bit": True,
                "batchSize": 2,
                "gradientAccumulationSteps": 4,
                "warmupSteps": 10,
                "maxSteps": 60,
                "learningRate": 0.0002,
                "loggingSteps": 1,
                "optimizer": "adamw_8bit",
                "weightDecay": 0.01,
                "lrSchedulerType": "linear",
                "seed": 42,
            }
        },
    }


def in_memory_dataset(rows):
    dataset = mock.MagicMock()
    rows = tuple(dict(row) for row in rows)
    dataset.column_names = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    dataset.__len__.return_value = len(rows)
    dataset.__iter__.side_effect = lambda: iter(rows)
    dataset.__getitem__.side_effect = rows.__getitem__

    projected_dataset = mock.MagicMock()

    def select_columns(column_names):
        projected_rows = tuple(
            {
                column_name: row[column_name]
                for column_name in column_names
                if column_name in row
            }
            for row in rows
        )
        projected_dataset.column_names = list(column_names)
        projected_dataset.__len__.return_value = len(projected_rows)
        projected_dataset.__iter__.side_effect = lambda: iter(projected_rows)
        projected_dataset.__getitem__.side_effect = projected_rows.__getitem__
        return projected_dataset

    dataset.select_columns.side_effect = select_columns
    dataset.projected_dataset = projected_dataset
    return dataset


def preprocessing_tokenizer():
    tokenizer = mock.Mock(
        eos_token="<eos>",
        eos_token_id=2,
        bos_token="<bos>",
        chat_template="",
    )

    def tokenize(text, *, add_special_tokens):
        if text == target_unsloth.PREPROCESSING_VERIFICATION_PROMPT:
            return {"input_ids": [11, 12]}
        if text == (
            target_unsloth.PREPROCESSING_VERIFICATION_PROMPT
            + target_unsloth.PREPROCESSING_VERIFICATION_COMPLETION
            + tokenizer.eos_token
        ):
            return {"input_ids": [11, 12, 13, tokenizer.eos_token_id]}
        raise AssertionError(f"unexpected verification text {text!r}")

    tokenizer.side_effect = tokenize
    return tokenizer


def example_train_dependencies(dataset):
    base_model = mock.Mock()
    adapter_model = mock.Mock()
    adapter_model.peft_config = {"default": mock.Mock()}
    tokenizer = preprocessing_tokenizer()
    fast_language_model = mock.Mock()
    fast_language_model.from_pretrained.return_value = (base_model, tokenizer)
    fast_language_model.get_peft_model.return_value = adapter_model
    trainer = mock.Mock()
    trainer.args = SimpleNamespace(
        packing=False,
        max_length=512,
        dataset_num_proc=2,
    )
    dependencies = target_unsloth.TrainDependencies(
        fast_language_model=fast_language_model,
        is_bfloat16_supported=mock.Mock(return_value=True),
        dataset_from_dict=mock.Mock(),
        load_dataset=mock.Mock(return_value=dataset),
        model_info=mock.Mock(return_value=mock.Mock(sha="a" * 40)),
        resolve_model_name=mock.Mock(return_value="example/resolved-model"),
        sft_config=mock.Mock(return_value="sft-config"),
        sft_trainer=mock.Mock(return_value=trainer),
    )
    return dependencies


def example_export_config():
    return {
        "baseModel": "example/model",
        "config": {
            "unsloth": {
                "maxSeqLength": 2048,
                "loadIn4bit": True,
            }
        },
        "output": {"quantize": "q4_k_m"},
    }


class ConfigTest(unittest.TestCase):
    def test_parse_config_uses_injected_loader(self):
        expected = {"baseModel": "example/model"}

        self.assertEqual(
            target_unsloth.parse_config(json.dumps(expected), loader=json.loads),
            expected,
        )

    def test_parse_config_rejects_non_mapping_roots(self):
        cases = (
            ("", lambda _: None),
            ("null", json.loads),
            ("[]", json.loads),
            ('"value"', json.loads),
        )

        for config_text, loader in cases:
            with self.subTest(config_text=config_text):
                with self.assertRaisesRegex(
                    ValueError,
                    "configuration root must be a mapping",
                ):
                    target_unsloth.parse_config(config_text, loader=loader)

    def test_load_config_reads_utf8_content(self):
        expected = {"baseModel": "organization/mödel"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(
                target_unsloth.load_config(config_path, loader=json.loads),
                expected,
            )

    def test_phase_paths_match_the_mount_contract(self):
        self.assertEqual(
            target_unsloth.TRAIN_CONFIG_PATH,
            Path("/aikit-config/train-config.yaml"),
        )
        self.assertEqual(
            target_unsloth.EXPORT_CONFIG_PATH,
            Path("/aikit-config/export-config.yaml"),
        )
        self.assertEqual(
            target_unsloth.TRAINED_MODEL_DIRECTORY,
            Path("/aikit-trained-model"),
        )
        self.assertEqual(target_unsloth.ARTIFACT_DIRECTORY, Path("/model"))

    def test_output_config_requires_nested_output(self):
        export_config = example_export_config()

        self.assertIs(
            target_unsloth.output_config(export_config),
            export_config["output"],
        )


class CLITest(unittest.TestCase):
    def test_train_mode_loads_only_train_config_without_logging_values(self):
        train_config = example_train_config()
        train_config["datasets"][0]["source"] = (
            "https://example.test/train.json?query=must-not-log"
        )
        output = io.StringIO()

        with (
            mock.patch.object(
                target_unsloth,
                "load_config",
                return_value=train_config,
            ) as load_config,
            mock.patch.object(target_unsloth, "train_model") as train_model,
            redirect_stdout(output),
        ):
            target_unsloth.main(["train"])

        load_config.assert_called_once_with(target_unsloth.TRAIN_CONFIG_PATH)
        train_model.assert_called_once_with(train_config)
        self.assertEqual(output.getvalue(), "Loaded fine-tuning configuration.\n")
        self.assertNotIn("must-not-log", output.getvalue())

    def test_export_mode_loads_only_export_config_without_logging_values(self):
        export_config = example_export_config()
        export_config["baseModel"] = "private/model?query=must-not-log"
        output = io.StringIO()

        with (
            mock.patch.object(
                target_unsloth,
                "load_config",
                return_value=export_config,
            ) as load_config,
            mock.patch.object(target_unsloth, "export_model") as export_model,
            redirect_stdout(output),
        ):
            target_unsloth.main(["export"])

        load_config.assert_called_once_with(target_unsloth.EXPORT_CONFIG_PATH)
        export_model.assert_called_once_with(export_config)
        self.assertEqual(output.getvalue(), "Loaded export configuration.\n")
        self.assertNotIn("must-not-log", output.getvalue())


class DatasetSourceTest(unittest.TestCase):
    def test_classifies_http_and_https_urls_as_json(self):
        for source in (
            "http://example.test/train.json",
            "https://example.test/train.json?query=opaque-marker",
            "HTTPS://example.test/train.json",
        ):
            with self.subTest(source=source):
                self.assertIs(
                    target_unsloth.classify_dataset_source(source),
                    target_unsloth.DatasetSourceKind.JSON_URL,
                )

    def test_classifies_dataset_identifiers_and_local_paths_as_datasets(self):
        for source in (
            "organization/dataset",
            "/datasets/train.json",
            "http-dataset",
            "ftp://example.test/train.json",
        ):
            with self.subTest(source=source):
                self.assertIs(
                    target_unsloth.classify_dataset_source(source),
                    target_unsloth.DatasetSourceKind.DATASET,
                )

    def test_builds_remote_json_load_spec_without_altering_url(self):
        source = "https://example.test/train.json?query=opaque-marker"

        self.assertEqual(
            target_unsloth.dataset_load_spec(source),
            target_unsloth.DatasetLoadSpec(
                path="json",
                kwargs={"data_files": {"train": source}, "split": "train"},
            ),
        )

    def test_builds_hub_dataset_load_spec(self):
        self.assertEqual(
            target_unsloth.dataset_load_spec("organization/dataset"),
            target_unsloth.DatasetLoadSpec(
                path="organization/dataset",
                kwargs={"split": "train"},
            ),
        )


class AlpacaFormattingTest(unittest.TestCase):
    def test_formats_batched_alpaca_examples(self):
        examples = {
            "instruction": ["Summarize", "Translate"],
            "input": ["A long passage", "Hello"],
            "output": ["A summary", "Bonjour"],
        }

        formatted = target_unsloth.format_alpaca_examples(
            examples,
            end_of_sequence="<eos>",
        )

        self.assertEqual(
            formatted,
            {
                "text": [
                    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize

### Input:
A long passage

### Response:
A summary<eos>""",
                    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate

### Input:
Hello

### Response:
Bonjour<eos>""",
                ]
            },
        )
        self.assertEqual(examples["instruction"], ["Summarize", "Translate"])


class PromptCompletionPreprocessingContractTest(unittest.TestCase):
    def test_verifies_prompt_mask_completion_labels_and_eos(self):
        prepared_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer = mock.Mock()
        trainer._prepare_dataset.return_value = [prepared_record]
        trainer.data_collator.return_value = {
            "input_ids": [[11, 12, 13, 2]],
            "labels": [[-100, -100, 13, 2]],
            "position_ids": [[0, 1, 2, 3]],
        }
        trainer.args = SimpleNamespace(
            packing=True,
            max_length=1,
            dataset_num_proc=2,
        )
        dataset_from_dict = mock.Mock(return_value="verification-dataset")
        processing_class = preprocessing_tokenizer()

        target_unsloth.verify_prompt_completion_preprocessing(
            trainer,
            dataset_from_dict=dataset_from_dict,
            processing_class=processing_class,
        )

        dataset_from_dict.assert_called_once_with(
            {
                "prompt": [target_unsloth.PREPROCESSING_VERIFICATION_PROMPT],
                "completion": [
                    target_unsloth.PREPROCESSING_VERIFICATION_COMPLETION
                ],
            }
        )
        trainer._prepare_dataset.assert_called_once_with(
            "verification-dataset",
            processing_class,
            mock.ANY,
            True,
            None,
            "prompt-completion verification",
        )
        verification_args = trainer._prepare_dataset.call_args.args[2]
        self.assertIsNot(verification_args, trainer.args)
        self.assertEqual(verification_args.max_length, 4)
        self.assertEqual(verification_args.dataset_num_proc, 1)
        self.assertEqual(trainer.args.max_length, 1)
        self.assertEqual(trainer.args.dataset_num_proc, 2)
        trainer.data_collator.assert_called_once_with([prepared_record])

    def test_rejects_incorrect_prompt_completion_labels_or_eos(self):
        cases = (
            (
                "prompt token is supervised",
                [11, 12, 13, 2],
                [0, 0, 1, 1],
                [11, -100, 13, 2],
                "prompt tokens must be masked",
            ),
            (
                "completion token is masked",
                [11, 12, 13, 2],
                [0, 0, 1, 1],
                [-100, -100, -100, 2],
                "completion tokens must be supervised",
            ),
            (
                "completion mask boundary is too early",
                [11, 12, 13, 2],
                [0, 1, 1, 1],
                [-100, 12, 13, 2],
                "boundary does not match",
            ),
            (
                "completion mask has no completion",
                [11, 12, 2],
                [0, 0, 0],
                [-100, -100, -100],
                "must identify prompt and completion tokens",
            ),
            (
                "terminal token is not eos",
                [11, 12, 13, 14],
                [0, 0, 1, 1],
                [-100, -100, 13, 14],
                "must end with the tokenizer EOS token",
            ),
        )

        for name, input_ids, completion_mask, labels, error_pattern in cases:
            with self.subTest(name=name):
                trainer = mock.Mock()
                trainer.args = SimpleNamespace(
                    packing=False,
                    max_length=512,
                    dataset_num_proc=2,
                )
                trainer._prepare_dataset.return_value = [
                    {
                        "input_ids": input_ids,
                        "completion_mask": completion_mask,
                    }
                ]
                trainer.data_collator.return_value = {
                    "input_ids": [input_ids],
                    "labels": [labels],
                }

                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.verify_prompt_completion_preprocessing(
                        trainer,
                        dataset_from_dict=mock.Mock(
                            return_value="verification-dataset"
                        ),
                        processing_class=preprocessing_tokenizer(),
                    )


class TrainingPhaseTest(unittest.TestCase):
    def assert_dataset_rejected_before_model(
        self,
        *,
        dataset_type,
        rows,
        error_pattern,
    ):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = dataset_type
        dataset = in_memory_dataset(rows)
        dependencies = example_train_dependencies(dataset)

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.train_model(
                        train_config,
                        trained_model_directory=Path(temporary_directory)
                        / "trained-model",
                        dependencies=dependencies,
                    )
        finally:
            dependencies.fast_language_model.from_pretrained.assert_not_called()
            dependencies.fast_language_model.get_peft_model.assert_not_called()
            dependencies.resolve_model_name.assert_not_called()
            dependencies.model_info.assert_not_called()
            dependencies.sft_trainer.assert_not_called()

        dependencies.load_dataset.assert_called_once_with(
            "organization/dataset",
            split="train",
        )

    def test_trains_and_saves_adapter_and_tokenizer(self):
        train_config = example_train_config()
        dataset = in_memory_dataset(
            [
                {
                    "instruction": "Summarize",
                    "input": "A long passage",
                    "output": "A summary",
                    "prompt": "must not change dataset dispatch",
                    "completion": "must not change dataset dispatch",
                }
            ]
        )
        mapped_dataset = mock.Mock()
        dataset.projected_dataset.map.return_value = mapped_dataset
        dependencies = example_train_dependencies(dataset)
        fast_language_model = dependencies.fast_language_model
        base_model = fast_language_model.from_pretrained.return_value[0]
        tokenizer = fast_language_model.from_pretrained.return_value[1]
        adapter_model = fast_language_model.get_peft_model.return_value
        adapter_config = adapter_model.peft_config["default"]
        trainer = dependencies.sft_trainer.return_value

        with tempfile.TemporaryDirectory() as temporary_directory:
            trained_model_directory = Path(temporary_directory) / "trained-model"
            result = target_unsloth.train_model(
                train_config,
                trained_model_directory=trained_model_directory,
                dependencies=dependencies,
            )

        self.assertEqual(result, trained_model_directory)
        fast_language_model.from_pretrained.assert_called_once_with(
            model_name="example/model",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        fast_language_model.get_peft_model.assert_called_once_with(
            base_model,
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
            random_state=42,
            use_rslora=False,
            loftq_config=None,
            base_model_name_or_path="example/resolved-model",
            revision="a" * 40,
        )
        dependencies.resolve_model_name.assert_called_once_with(
            "example/model",
            load_in_4bit=False,
        )
        dependencies.model_info.assert_called_once_with(
            repo_id="example/resolved-model"
        )
        self.assertEqual(
            adapter_config.base_model_name_or_path,
            "example/resolved-model",
        )
        self.assertEqual(adapter_config.revision, "a" * 40)
        dependencies.load_dataset.assert_called_once_with(
            "organization/dataset",
            split="train",
        )
        dataset.select_columns.assert_called_once_with(
            ["instruction", "input", "output"]
        )
        dataset.map.assert_not_called()
        self.assertTrue(
            dataset.projected_dataset.map.call_args.kwargs["batched"]
        )
        dependencies.sft_trainer.assert_called_once_with(
            model=adapter_model,
            train_dataset=mapped_dataset,
            processing_class=tokenizer,
            args="sft-config",
        )
        trainer.train.assert_called_once_with()
        adapter_model.save_pretrained.assert_called_once_with(
            trained_model_directory
        )
        tokenizer.save_pretrained.assert_called_once_with(
            trained_model_directory
        )
        self.assertFalse(dependencies.sft_config.call_args.kwargs["fp16"])
        self.assertTrue(dependencies.sft_config.call_args.kwargs["bf16"])
        self.assertIs(
            dependencies.sft_config.call_args.kwargs.get("completion_only_loss"),
            False,
        )

    def test_prompt_completion_dataset_is_passed_through_with_completion_loss(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "prompt-completion"
        rows = [
            {
                "prompt": "Question: What is a container image?\nAnswer:",
                "completion": " An immutable application package.",
                "input_ids": [1, 2, 3],
            },
            {
                "prompt": "",
                "completion": " A prompt may be empty.",
                "labels": [4, 5, 6],
            },
        ]
        dataset = in_memory_dataset(rows)
        dependencies = example_train_dependencies(dataset)
        verification_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer = dependencies.sft_trainer.return_value
        trainer._prepare_dataset.return_value = [verification_record]
        trainer.data_collator.return_value = {
            "input_ids": [verification_record["input_ids"]],
            "labels": [[-100, -100, 13, 2]],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        dataset.select_columns.assert_called_once_with(["prompt", "completion"])
        dataset.map.assert_not_called()
        dataset.projected_dataset.map.assert_not_called()
        self.assertIs(
            dependencies.sft_trainer.call_args.kwargs["train_dataset"],
            dataset.projected_dataset,
        )
        self.assertEqual(
            list(dataset.projected_dataset),
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                }
                for row in rows
            ],
        )
        self.assertIs(
            dependencies.sft_config.call_args.kwargs.get("completion_only_loss"),
            True,
        )
        dependencies.dataset_from_dict.assert_called_once_with(
            {
                "prompt": [target_unsloth.PREPROCESSING_VERIFICATION_PROMPT],
                "completion": [
                    target_unsloth.PREPROCESSING_VERIFICATION_COMPLETION
                ],
            }
        )
        trainer.data_collator.assert_called_once_with([verification_record])

    def test_rejects_unknown_dataset_type_before_loading_or_model_allocation(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "other"
        dataset = in_memory_dataset(
            [{"instruction": "Do something", "input": "", "output": "Done"}]
        )
        dependencies = example_train_dependencies(dataset)

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                with self.assertRaisesRegex(ValueError, "unsupported dataset type"):
                    target_unsloth.train_model(
                        train_config,
                        trained_model_directory=Path(temporary_directory)
                        / "trained-model",
                        dependencies=dependencies,
                    )
        finally:
            dependencies.load_dataset.assert_not_called()
            dependencies.fast_language_model.from_pretrained.assert_not_called()
            dependencies.fast_language_model.get_peft_model.assert_not_called()
            dependencies.resolve_model_name.assert_not_called()
            dependencies.model_info.assert_not_called()
            dependencies.sft_trainer.assert_not_called()

    def test_rejects_empty_or_invalid_datasets_before_model_allocation(self):
        valid_prompt_completion = {
            "prompt": "Question?",
            "completion": " Answer.",
        }
        cases = (
            (
                "empty prompt-completion dataset",
                "prompt-completion",
                [],
                "empty|at least one",
            ),
            (
                "missing prompt-completion column",
                "prompt-completion",
                [{"prompt": "Question?"}],
                "completion",
            ),
            (
                "null prompt",
                "prompt-completion",
                [valid_prompt_completion, {"prompt": None, "completion": " Answer."}],
                "prompt.*string",
            ),
            (
                "non-string prompt",
                "prompt-completion",
                [valid_prompt_completion, {"prompt": 42, "completion": " Answer."}],
                "prompt.*string",
            ),
            (
                "null completion",
                "prompt-completion",
                [valid_prompt_completion, {"prompt": "Question?", "completion": None}],
                "completion.*string",
            ),
            (
                "non-string completion",
                "prompt-completion",
                [valid_prompt_completion, {"prompt": "Question?", "completion": []}],
                "completion.*string",
            ),
            (
                "empty completion",
                "prompt-completion",
                [valid_prompt_completion, {"prompt": "Question?", "completion": ""}],
                "completion.*non-empty string",
            ),
            (
                "missing Alpaca column",
                "alpaca",
                [{"instruction": "Summarize", "input": "Passage"}],
                "output",
            ),
            (
                "null Alpaca field",
                "alpaca",
                [{"instruction": None, "input": "", "output": "Summary"}],
                "instruction.*string",
            ),
            (
                "non-string Alpaca field",
                "alpaca",
                [{"instruction": "Summarize", "input": 42, "output": "Summary"}],
                "input.*string",
            ),
        )

        for name, dataset_type, rows, error_pattern in cases:
            with self.subTest(name=name):
                self.assert_dataset_rejected_before_model(
                    dataset_type=dataset_type,
                    rows=rows,
                    error_pattern=error_pattern,
                )

    def test_rejects_training_when_export_base_revision_is_not_immutable(self):
        train_config = example_train_config()
        base_model = mock.Mock()
        fast_language_model = mock.Mock()
        fast_language_model.from_pretrained.return_value = (
            base_model,
            mock.Mock(),
        )
        dataset = in_memory_dataset(
            [
                {
                    "instruction": "Summarize",
                    "input": "A long passage",
                    "output": "A summary",
                }
            ]
        )
        dependencies = target_unsloth.TrainDependencies(
            fast_language_model=fast_language_model,
            is_bfloat16_supported=mock.Mock(),
            dataset_from_dict=mock.Mock(),
            load_dataset=mock.Mock(return_value=dataset),
            model_info=mock.Mock(return_value=mock.Mock(sha=None)),
            resolve_model_name=mock.Mock(return_value="example/resolved-model"),
            sft_config=mock.Mock(),
            sft_trainer=mock.Mock(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "resolved export base model revision is not an immutable",
        ):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        fast_language_model.get_peft_model.assert_not_called()
        dependencies.load_dataset.assert_called_once_with(
            "organization/dataset",
            split="train",
        )
        dataset.map.assert_not_called()
        dataset.projected_dataset.map.assert_not_called()


class ExportPhaseTest(unittest.TestCase):
    def test_reloads_adapter_and_exports_staged_gguf(self):
        export_config = example_export_config()
        model = mock.Mock()
        tokenizer = mock.Mock()
        fast_language_model = mock.Mock()
        fast_language_model.from_pretrained.return_value = (model, tokenizer)
        snapshot_download = mock.Mock()
        dependencies = target_unsloth.ExportDependencies(
            fast_language_model=fast_language_model,
            snapshot_download=snapshot_download,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trained_model_directory = root / "trained-model"
            export_directory = root / "export"
            generated_directory = root / "export_gguf"
            artifact_directory = root / "model"
            trained_model_directory.mkdir()
            adapter_config_path = (
                trained_model_directory / target_unsloth.ADAPTER_CONFIG_FILENAME
            )
            adapter_config_path.write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "example/resolved-model",
                        "revision": "a" * 40,
                    }
                ),
                encoding="utf-8",
            )
            snapshot_path = root / "cache" / "snapshots" / ("a" * 40)
            snapshot_download.return_value = str(snapshot_path)
            generated_directory.mkdir()
            gguf_file = generated_directory / "model-q4_k_m.gguf"
            gguf_file.write_bytes(b"gguf")
            model.save_pretrained_gguf.return_value = {
                "gguf_files": [str(gguf_file)]
            }

            staged_file = target_unsloth.export_model(
                export_config,
                trained_model_directory=trained_model_directory,
                export_directory=export_directory,
                artifact_directory=artifact_directory,
                dependencies=dependencies,
            )

            self.assertEqual(staged_file, artifact_directory / gguf_file.name)
            self.assertEqual(staged_file.read_bytes(), b"gguf")
            self.assertFalse(generated_directory.exists())
            pinned_adapter_config = json.loads(
                adapter_config_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                pinned_adapter_config["base_model_name_or_path"],
                str(snapshot_path),
            )

        snapshot_download.assert_called_once_with(
            repo_id="example/resolved-model",
            revision="a" * 40,
        )
        fast_language_model.from_pretrained.assert_called_once_with(
            model_name=str(trained_model_directory),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            local_files_only=True,
        )
        model.save_pretrained_gguf.assert_called_once_with(
            export_directory,
            tokenizer,
            quantization_method="q4_k_m",
        )

    def test_rejects_export_when_saved_revision_is_mutable(self):
        dependencies = target_unsloth.ExportDependencies(
            fast_language_model=mock.Mock(),
            snapshot_download=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trained_model_directory = Path(temporary_directory) / "trained-model"
            trained_model_directory.mkdir()
            (
                trained_model_directory / target_unsloth.ADAPTER_CONFIG_FILENAME
            ).write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "example/model",
                        "revision": "main",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "saved adapter base model revision is not an immutable",
            ):
                target_unsloth.export_model(
                    example_export_config(),
                    trained_model_directory=trained_model_directory,
                    dependencies=dependencies,
                )

        dependencies.snapshot_download.assert_not_called()
        dependencies.fast_language_model.from_pretrained.assert_not_called()


class GGUFResultTest(unittest.TestCase):
    def test_validates_single_gguf_output(self):
        self.assertEqual(
            target_unsloth.validate_gguf_result(
                {"gguf_files": ["/exports/model-q4_k_m.gguf"]}
            ),
            Path("/exports/model-q4_k_m.gguf"),
        )

    def test_rejects_missing_or_multiple_gguf_outputs(self):
        for gguf_files in ([], ["one.gguf", "two.gguf"]):
            with self.subTest(gguf_files=gguf_files):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "expected exactly one GGUF output",
                ):
                    target_unsloth.validate_gguf_result(
                        {"gguf_files": gguf_files}
                    )

    def test_staged_path_uses_the_generated_filename(self):
        self.assertEqual(
            target_unsloth.staged_gguf_path(
                "/exports/model-q4_k_m.gguf",
                "/model",
            ),
            Path("/model/model-q4_k_m.gguf"),
        )

    def test_stages_artifact_and_cleans_export_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            export_directory = root / "export"
            generated_directory = root / "export_gguf"
            artifact_directory = root / "model"
            export_directory.mkdir()
            generated_directory.mkdir()
            gguf_file = generated_directory / "model-q4_k_m.gguf"
            gguf_file.write_bytes(b"gguf")
            (export_directory / "temporary-file").write_text(
                "temporary",
                encoding="utf-8",
            )

            staged_file = target_unsloth.stage_gguf_artifact(
                gguf_file,
                artifact_directory,
            )
            target_unsloth.cleanup_gguf_export(export_directory)

            self.assertEqual(staged_file, artifact_directory / gguf_file.name)
            self.assertEqual(staged_file.read_bytes(), b"gguf")
            self.assertFalse(gguf_file.exists())
            self.assertFalse(export_directory.exists())
            self.assertFalse(generated_directory.exists())


if __name__ == "__main__":
    unittest.main()
