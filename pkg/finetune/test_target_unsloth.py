import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("target_unsloth.py")
MODULE_SPEC = importlib.util.spec_from_file_location("target_unsloth", MODULE_PATH)
target_unsloth = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(target_unsloth)


def example_train_config():
    return {
        "baseModel": "example/model",
        "datasets": [{"source": "organization/dataset"}],
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
                    target_unsloth.ALPACA_PROMPT.format(
                        "Summarize", "A long passage", "A summary"
                    )
                    + "<eos>",
                    target_unsloth.ALPACA_PROMPT.format(
                        "Translate", "Hello", "Bonjour"
                    )
                    + "<eos>",
                ]
            },
        )
        self.assertEqual(examples["instruction"], ["Summarize", "Translate"])


class TrainingPhaseTest(unittest.TestCase):
    def test_trains_and_saves_adapter_and_tokenizer(self):
        train_config = example_train_config()
        base_model = mock.Mock()
        adapter_model = mock.Mock()
        tokenizer = mock.Mock(eos_token="<eos>")
        fast_language_model = mock.Mock()
        fast_language_model.from_pretrained.return_value = (base_model, tokenizer)
        fast_language_model.get_peft_model.return_value = adapter_model
        dataset = mock.Mock()
        mapped_dataset = mock.Mock()
        dataset.map.return_value = mapped_dataset
        load_dataset = mock.Mock(return_value=dataset)
        model_info = mock.Mock(return_value=mock.Mock(sha="a" * 40))
        resolve_model_name = mock.Mock(return_value="example/resolved-model")
        trainer = mock.Mock()
        sft_trainer = mock.Mock(return_value=trainer)
        sft_config = mock.Mock(return_value="sft-config")
        dependencies = target_unsloth.TrainDependencies(
            fast_language_model=fast_language_model,
            is_bfloat16_supported=mock.Mock(return_value=True),
            load_dataset=load_dataset,
            model_info=model_info,
            resolve_model_name=resolve_model_name,
            sft_config=sft_config,
            sft_trainer=sft_trainer,
        )

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
        resolve_model_name.assert_called_once_with(
            "example/model",
            load_in_4bit=False,
        )
        model_info.assert_called_once_with(repo_id="example/resolved-model")
        load_dataset.assert_called_once_with(
            "organization/dataset",
            split="train",
        )
        self.assertTrue(dataset.map.call_args.kwargs["batched"])
        sft_trainer.assert_called_once_with(
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
        self.assertFalse(sft_config.call_args.kwargs["fp16"])
        self.assertTrue(sft_config.call_args.kwargs["bf16"])

    def test_rejects_training_when_export_base_revision_is_not_immutable(self):
        train_config = example_train_config()
        base_model = mock.Mock()
        fast_language_model = mock.Mock()
        fast_language_model.from_pretrained.return_value = (
            base_model,
            mock.Mock(),
        )
        dependencies = target_unsloth.TrainDependencies(
            fast_language_model=fast_language_model,
            is_bfloat16_supported=mock.Mock(),
            load_dataset=mock.Mock(),
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
        dependencies.load_dataset.assert_not_called()


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
