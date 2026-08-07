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

    def token_ids(text):
        if text == target_unsloth.PREPROCESSING_VERIFICATION_PROMPT:
            return [11, 12]
        if text == (
            target_unsloth.PREPROCESSING_VERIFICATION_PROMPT
            + target_unsloth.PREPROCESSING_VERIFICATION_COMPLETION
            + tokenizer.eos_token
        ):
            return [11, 12, 13, tokenizer.eos_token_id]
        if text.endswith(tokenizer.eos_token):
            return [11, 12, 13, tokenizer.eos_token_id]
        return [11, 12]

    def tokenize(text, *, add_special_tokens):
        if isinstance(text, list):
            return {"input_ids": [token_ids(item) for item in text]}
        return {"input_ids": token_ids(text)}

    tokenizer.side_effect = tokenize
    return tokenizer


class TextPreprocessingTokenizer:
    def __init__(
        self,
        *,
        auto_bos=True,
        auto_eos=False,
        bos_token="<bos>",
        bos_token_id=1,
        eos_token="<eos>",
        eos_token_id=2,
        chat_template="",
    ):
        self.auto_bos = auto_bos
        self.auto_eos = auto_eos
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.chat_template = chat_template
        self.save_pretrained = mock.Mock()
        self.calls = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        if isinstance(text, list):
            return {
                "input_ids": [
                    self.token_ids(
                        item,
                        add_special_tokens=add_special_tokens,
                        truncation=truncation,
                        max_length=max_length,
                    )
                    for item in text
                ]
            }

        return {
            "input_ids": self.token_ids(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
        }

    def token_ids(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
        max_length,
    ):
        input_ids = []
        offset = 0
        special_tokens = tuple(
            (token, token_id)
            for token, token_id in (
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            )
            if token is not None
        )
        while offset < len(text):
            for token, token_id in special_tokens:
                if text.startswith(token, offset):
                    input_ids.append(token_id)
                    offset += len(token)
                    break
            else:
                input_ids.append(1000 + ord(text[offset]))
                offset += 1

        if add_special_tokens and self.auto_bos and self.bos_token_id is not None:
            input_ids.insert(0, self.bos_token_id)
        if add_special_tokens and self.auto_eos and self.eos_token_id is not None:
            input_ids.append(self.eos_token_id)
        if truncation:
            if max_length is None:
                raise AssertionError("truncated tokenization requires max_length")
            input_ids = input_ids[:max_length]

        return input_ids


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

    def test_describes_remote_json_source_without_credentials(self):
        source = (
            "https://alice:password@example.test/private/train.json?"
            "X-Amz-Credential=credential-marker&X-Amz-Signature=signature-marker"
            "#fragment-marker"
        )

        subject = target_unsloth.dataset_error_subject(
            target_unsloth.DATASET_TYPE_TEXT,
            source=source,
            record_index=7,
        )

        self.assertEqual(subject, "text dataset remote JSON URL row 7")
        for secret in (
            source,
            "alice",
            "password",
            "X-Amz-Credential",
            "credential-marker",
            "X-Amz-Signature",
            "signature-marker",
            "fragment-marker",
        ):
            self.assertNotIn(secret, subject)

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


class TextBoundaryNormalizationTest(unittest.TestCase):
    def test_uses_tokenizer_added_bos_and_one_explicit_eos(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)

        normalized = target_unsloth.normalize_text_examples(
            {
                "text": [
                    "Preserve <bos> and <eos> inside the body.",
                    "<bos><bos>Already bounded.<eos><eos>",
                ]
            },
            policy=policy,
        )

        self.assertTrue(policy.add_special_tokens)
        self.assertEqual(
            normalized,
            {
                "text": [
                    "Preserve <bos> and <eos> inside the body.<eos>",
                    "Already bounded.<eos>",
                ]
            },
        )
        for text in normalized["text"]:
            input_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            self.assertEqual(
                target_unsloth.leading_token_count(input_ids, tokenizer.bos_token_id),
                1,
            )
            self.assertEqual(
                target_unsloth.trailing_token_count(input_ids, tokenizer.eos_token_id),
                1,
            )

    def test_uses_explicit_bos_when_tokenizer_does_not_add_one(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=False)
        policy = target_unsloth.text_boundary_policy(tokenizer)

        normalized = target_unsloth.normalize_text_examples(
            {"text": ["Missing boundaries.", "<bos>Existing BOS.<eos>"]},
            policy=policy,
        )

        self.assertFalse(policy.add_special_tokens)
        self.assertEqual(
            normalized,
            {
                "text": [
                    "<bos>Missing boundaries.<eos>",
                    "<bos>Existing BOS.<eos>",
                ]
            },
        )

    def test_uses_explicit_bos_when_chat_template_suppresses_special_tokens(self):
        tokenizer = TextPreprocessingTokenizer(
            auto_bos=True,
            chat_template="{{ '<bos>' }}{{ messages }}",
        )
        policy = target_unsloth.text_boundary_policy(tokenizer)

        normalized = target_unsloth.normalize_text_value(
            "Content.",
            policy=policy,
        )

        self.assertFalse(policy.add_special_tokens)
        self.assertEqual(normalized, "<bos>Content.<eos>")

    def test_supports_tokenizer_without_bos(self):
        tokenizer = TextPreprocessingTokenizer(
            auto_bos=False,
            bos_token=None,
            bos_token_id=None,
        )
        policy = target_unsloth.text_boundary_policy(tokenizer)

        normalized = target_unsloth.normalize_text_value(
            "Content.<eos>",
            policy=policy,
        )

        self.assertIsNone(policy.bos_token)
        self.assertTrue(policy.append_eos_token)
        self.assertEqual(normalized, "Content.<eos>")
        self.assertEqual(
            tokenizer(normalized, add_special_tokens=True)["input_ids"][-1],
            tokenizer.eos_token_id,
        )

    def test_uses_automatic_eos_for_tokenizer_without_bos(self):
        tokenizer = TextPreprocessingTokenizer(
            auto_bos=False,
            auto_eos=True,
            bos_token=None,
            bos_token_id=None,
        )

        policy = target_unsloth.text_boundary_policy(tokenizer)
        normalized = target_unsloth.normalize_text_value(
            "Content.<eos><eos>",
            policy=policy,
        )
        input_ids = tokenizer(
            normalized,
            add_special_tokens=policy.add_special_tokens,
        )["input_ids"]

        self.assertTrue(policy.add_special_tokens)
        self.assertFalse(policy.append_eos_token)
        self.assertEqual(normalized, "Content.")
        self.assertEqual(
            target_unsloth.trailing_token_count(
                input_ids,
                tokenizer.eos_token_id,
            ),
            1,
        )

    def test_rejects_missing_or_unusable_eos(self):
        cases = (
            (
                TextPreprocessingTokenizer(eos_token=None, eos_token_id=None),
                "non-empty tokenizer EOS token",
            ),
            (
                TextPreprocessingTokenizer(eos_token_id=None),
                "integer tokenizer EOS token ID",
            ),
        )

        for tokenizer, error_pattern in cases:
            with self.subTest(error_pattern=error_pattern):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.text_boundary_policy(tokenizer)

    def test_accepts_exact_token_budget_and_rejects_overflow(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)

        target_unsloth.validate_text_sequence_lengths(
            [{"text": "<bos>ABC<eos><eos>"}],
            processing_class=tokenizer,
            policy=policy,
            max_seq_length=5,
            source="organization/dataset",
        )

        with self.assertRaisesRegex(
            ValueError,
            "text dataset source 'organization/dataset' row 1.*at least 5 tokens.*maxSeqLength 4",
        ):
            target_unsloth.validate_text_sequence_lengths(
                [{"text": "AB"}, {"text": "<bos>ABC<eos><eos>"}],
                processing_class=tokenizer,
                policy=policy,
                max_seq_length=4,
                source="organization/dataset",
            )

    def test_validates_in_bounded_truncated_batches(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        tokenizer.calls.clear()
        rows = [
            {"text": "A"},
            {"text": "B"},
            {"text": "C"},
            {"text": "D"},
            {"text": "overflow"},
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"row 4.*at least 5 tokens.*maxSeqLength 4",
        ):
            target_unsloth.validate_text_sequence_lengths(
                rows,
                processing_class=tokenizer,
                policy=policy,
                max_seq_length=4,
                source="organization/dataset",
                batch_size=2,
            )

        self.assertEqual(
            [len(call["text"]) for call in tokenizer.calls],
            [2, 2, 1],
        )
        for call in tokenizer.calls:
            self.assertIsInstance(call["text"], list)
            self.assertTrue(call["truncation"])
            self.assertEqual(call["max_length"], 5)

    def test_limits_batches_to_bounded_token_budget(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        tokenizer.calls.clear()

        target_unsloth.validate_text_sequence_lengths(
            [{"text": f"row-{index}"} for index in range(65)],
            processing_class=tokenizer,
            policy=policy,
            max_seq_length=4095,
            source="organization/dataset",
        )

        self.assertEqual(
            [len(call["text"]) for call in tokenizer.calls],
            [64, 1],
        )

    def test_redacts_signed_url_in_text_validation_error(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        source = (
            "https://alice:password@example.test/private/train.json?"
            "X-Amz-Credential=credential-marker&X-Amz-Signature=signature-marker"
            "#fragment-marker"
        )

        with self.assertRaises(ValueError) as raised:
            target_unsloth.validate_text_sequence_lengths(
                [{"text": "A"}, {"text": "overflow"}],
                processing_class=tokenizer,
                policy=policy,
                max_seq_length=4,
                source=source,
            )

        message = str(raised.exception)
        self.assertIn("text dataset remote JSON URL row 1", message)
        self.assertIn("maxSeqLength 4", message)
        for secret in (
            source,
            "alice",
            "password",
            "X-Amz-Credential",
            "credential-marker",
            "X-Amz-Signature",
            "signature-marker",
            "fragment-marker",
        ):
            self.assertNotIn(secret, message)

    def test_validates_actual_record_boundaries(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        tokenizer.auto_eos = True

        with self.assertRaisesRegex(
            ValueError,
            "text dataset source 'organization/dataset' row 0.*one terminal.*EOS",
        ):
            target_unsloth.validate_text_sequence_lengths(
                [{"text": "Content."}],
                processing_class=tokenizer,
                policy=policy,
                max_seq_length=64,
                source="organization/dataset",
            )


class PromptCompletionPreprocessingContractTest(unittest.TestCase):
    def test_accepts_truncation_that_retains_supervised_eos(self):
        rows = [{"prompt": "prompt", "completion": " completion"}]
        tokenizer = mock.Mock(
            eos_token="<eos>",
            eos_token_id=2,
            bos_token="<bos>",
            chat_template="",
        )
        tokenizer.return_value = {
            "input_ids": [
                [11, 12, 99],
                [11, 12, 13, tokenizer.eos_token_id, tokenizer.eos_token_id],
            ]
        }

        fingerprint = target_unsloth.validate_prompt_completion_tokenization(
            rows,
            processing_class=tokenizer,
            max_seq_length=4,
        )

        expected_fingerprint = (
            target_unsloth.extend_prompt_prefix_fingerprint(
                target_unsloth.empty_prompt_prefix_fingerprint(),
                [11, 12, 13],
                description="expected retained prompt prefix",
            )
        )
        self.assertEqual(fingerprint, expected_fingerprint)

    def test_rejects_truncation_that_retains_no_completion(self):
        tokenizer = mock.Mock(
            eos_token="<eos>",
            eos_token_id=2,
            bos_token="<bos>",
            chat_template="",
        )
        tokenizer.return_value = {
            "input_ids": [
                [11, 12, 13, 14],
                [11, 12, 13, tokenizer.eos_token_id],
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError,
            r"record 0.*retains no completion tokens.*maxSeqLength 3",
        ):
            target_unsloth.validate_prompt_completion_tokenization(
                [{"prompt": "prompt", "completion": " completion"}],
                processing_class=tokenizer,
                max_seq_length=3,
            )

    def test_validates_source_boundaries_in_bounded_tokenizer_batches(self):
        rows = [
            {"prompt": f"prompt-{index}", "completion": f" completion-{index}"}
            for index in range(5)
        ]
        tokenizer = mock.Mock(
            eos_token="<eos>",
            eos_token_id=2,
            bos_token="<bos>",
            chat_template="",
        )

        def tokenize(texts, *, add_special_tokens):
            self.assertIsInstance(texts, list)
            self.assertLessEqual(len(texts), 4)
            token_rows = []
            for text in texts:
                record_index = int(text.split("-", 1)[1].split(" ", 1)[0])
                token_rows.append(
                    [record_index, record_index + 10, tokenizer.eos_token_id]
                    if text.endswith(tokenizer.eos_token)
                    else [record_index]
                )
            return {"input_ids": token_rows}

        tokenizer.side_effect = tokenize

        fingerprint = target_unsloth.validate_prompt_completion_tokenization(
            rows,
            processing_class=tokenizer,
            max_seq_length=3,
            batch_size=2,
        )

        self.assertEqual(
            [len(call.args[0]) for call in tokenizer.call_args_list],
            [4, 4, 2],
        )
        self.assertEqual(fingerprint.sequence_count, len(rows))

    def test_limits_default_batches_to_estimated_token_budget(self):
        rows = [
            {"prompt": f"prompt-{index}", "completion": " completion"}
            for index in range(65)
        ]
        tokenizer = mock.Mock(
            eos_token="<eos>",
            eos_token_id=2,
            bos_token="<bos>",
            chat_template="",
        )

        def tokenize(texts, *, add_special_tokens):
            token_rows = []
            for text in texts:
                record_index = int(text.split("-", 1)[1].split(" ", 1)[0])
                token_rows.append(
                    [record_index, tokenizer.eos_token_id]
                    if text.endswith(tokenizer.eos_token)
                    else [record_index]
                )
            return {"input_ids": token_rows}

        tokenizer.side_effect = tokenize

        target_unsloth.validate_prompt_completion_tokenization(
            rows,
            processing_class=tokenizer,
            max_seq_length=2048,
        )

        self.assertEqual(
            [len(call.args[0]) for call in tokenizer.call_args_list],
            [128, 2],
        )

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

    def test_disables_special_tokens_when_bos_is_in_chat_template(self):
        prepared_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer = mock.Mock()
        trainer.args = SimpleNamespace(
            packing=False,
            max_length=512,
            dataset_num_proc=2,
        )
        trainer._prepare_dataset.return_value = [prepared_record]
        trainer.data_collator.return_value = {
            "input_ids": [prepared_record["input_ids"]],
            "labels": [[-100, -100, 13, 2]],
        }
        processing_class = preprocessing_tokenizer()
        processing_class.chat_template = "<bos>{{ messages }}"

        target_unsloth.verify_prompt_completion_preprocessing(
            trainer,
            dataset_from_dict=mock.Mock(return_value="verification-dataset"),
            processing_class=processing_class,
        )

        self.assertEqual(
            processing_class.call_args_list,
            [
                mock.call(
                    target_unsloth.PREPROCESSING_VERIFICATION_PROMPT,
                    add_special_tokens=False,
                ),
                mock.call(
                    target_unsloth.PREPROCESSING_VERIFICATION_PROMPT
                    + target_unsloth.PREPROCESSING_VERIFICATION_COMPLETION
                    + processing_class.eos_token,
                    add_special_tokens=False,
                ),
            ],
        )

    def test_validates_multiple_bfd_packed_segments(self):
        target_unsloth.validate_prepared_prompt_completion_dataset(
            [
                {
                    "input_ids": [11, 13, 2, 21, 23, 2],
                    "completion_mask": [0, 1, 1, 0, 1, 1],
                    "seq_lengths": [3, 3],
                }
            ],
            eos_token_id=2,
            max_seq_length=6,
            packing=True,
        )

    def test_matches_reordered_duplicate_bfd_prompt_prefixes(self):
        expected_fingerprint = target_unsloth.empty_prompt_prefix_fingerprint()
        for index, prompt_ids in enumerate(([11], [21], [11])):
            expected_fingerprint = (
                target_unsloth.extend_prompt_prefix_fingerprint(
                    expected_fingerprint,
                    prompt_ids,
                    description=f"source prompt {index}",
                )
            )

        target_unsloth.validate_prepared_prompt_completion_dataset(
            [
                {
                    "input_ids": [21, 23, 2, 11, 13, 2, 11, 14, 2],
                    "completion_mask": [0, 1, 1, 0, 1, 1, 0, 1, 1],
                    "seq_lengths": [3, 3, 3],
                }
            ],
            eos_token_id=2,
            max_seq_length=9,
            packing=True,
            expected_prompt_prefix_fingerprint=expected_fingerprint,
        )

    def test_rejects_mutated_or_missing_prepared_prompt_prefixes(self):
        cases = (
            ("mutated", ([11],), [99]),
            ("missing", ([11], [21]), [11]),
        )

        for name, source_prompt_rows, prepared_prompt_ids in cases:
            with self.subTest(name=name):
                expected_fingerprint = (
                    target_unsloth.empty_prompt_prefix_fingerprint()
                )
                for index, prompt_ids in enumerate(source_prompt_rows):
                    expected_fingerprint = (
                        target_unsloth.extend_prompt_prefix_fingerprint(
                            expected_fingerprint,
                            prompt_ids,
                            description=f"source prompt {index}",
                        )
                    )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "prompt prefixes do not match",
                ):
                    target_unsloth.validate_prepared_prompt_completion_dataset(
                        [
                            {
                                "input_ids": prepared_prompt_ids + [13, 2],
                                "completion_mask": [0]
                                * len(prepared_prompt_ids)
                                + [1, 1],
                            }
                        ],
                        eos_token_id=2,
                        max_seq_length=4,
                        packing=False,
                        expected_prompt_prefix_fingerprint=expected_fingerprint,
                    )

    def test_rejects_wrapped_packing_validation(self):
        with self.assertRaisesRegex(RuntimeError, "requires the bfd packing strategy"):
            target_unsloth.validate_prepared_prompt_completion_dataset(
                [],
                eos_token_id=2,
                max_seq_length=6,
                packing=True,
                packing_strategy="wrapped",
            )


class TextPreprocessingContractTest(unittest.TestCase):
    def packed_verification_record(self, tokenizer, policy):
        normalized_texts = [
            target_unsloth.normalize_text_value(text, policy=policy)
            for text in target_unsloth.text_preprocessing_verification_sources(
                policy
            )
        ]
        segments = [
            tokenizer(
                text,
                add_special_tokens=policy.add_special_tokens,
            )["input_ids"]
            for text in normalized_texts
        ]
        segments.reverse()
        return normalized_texts, segments, {
            "input_ids": [input_id for segment in segments for input_id in segment],
            "seq_lengths": [len(segment) for segment in segments],
        }

    def test_verifies_packed_boundaries_and_full_sequence_labels(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        verification_sources = (
            target_unsloth.text_preprocessing_verification_sources(policy)
        )
        self.assertTrue(verification_sources[0].startswith("<bos><bos>"))
        self.assertTrue(verification_sources[0].endswith("<eos><eos>"))
        normalized_texts, segments, prepared_record = (
            self.packed_verification_record(tokenizer, policy)
        )
        self.assertFalse(normalized_texts[0].startswith("<bos>"))
        self.assertFalse(normalized_texts[0].endswith("<eos><eos>"))
        labels = list(prepared_record["input_ids"])
        offset = 0
        for segment in segments:
            labels[offset] = -100
            offset += len(segment)

        trainer = mock.Mock()
        trainer.args = SimpleNamespace(
            packing=True,
            max_length=1,
            dataset_num_proc=2,
        )
        trainer._prepare_dataset.return_value = [prepared_record]
        trainer.data_collator.return_value = {
            "input_ids": [prepared_record["input_ids"]],
            "labels": [labels],
        }
        dataset_from_dict = mock.Mock(return_value="verification-dataset")

        target_unsloth.verify_text_preprocessing(
            trainer,
            dataset_from_dict=dataset_from_dict,
            processing_class=tokenizer,
            policy=policy,
        )

        dataset_from_dict.assert_called_once_with({"text": normalized_texts})
        trainer._prepare_dataset.assert_called_once_with(
            "verification-dataset",
            tokenizer,
            mock.ANY,
            True,
            None,
            "text verification",
        )
        verification_args = trainer._prepare_dataset.call_args.args[2]
        self.assertEqual(
            verification_args.max_length,
            sum(prepared_record["seq_lengths"]),
        )
        self.assertEqual(verification_args.dataset_num_proc, 1)
        self.assertEqual(trainer.args.max_length, 1)
        self.assertEqual(trainer.args.dataset_num_proc, 2)
        trainer.data_collator.assert_called_once_with([prepared_record])

    def non_packed_verification(self, tokenizer, policy, *, mask_starts):
        normalized_texts = [
            target_unsloth.normalize_text_value(text, policy=policy)
            for text in target_unsloth.text_preprocessing_verification_sources(
                policy
            )
        ]
        prepared_records = [
            {
                "input_ids": tokenizer(
                    text,
                    add_special_tokens=policy.add_special_tokens,
                )["input_ids"]
            }
            for text in normalized_texts
        ]
        trainer = mock.Mock()
        trainer._prepare_dataset.return_value = prepared_records

        def collate(records):
            input_ids = records[0]["input_ids"]
            labels = list(input_ids)
            if mask_starts:
                labels[0] = -100
            return {"input_ids": [input_ids], "labels": [labels]}

        trainer.data_collator.side_effect = collate
        return trainer

    def test_verifies_non_packed_full_sequence_labels(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        trainer = self.non_packed_verification(
            tokenizer,
            policy,
            mask_starts=False,
        )
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            max_length=1,
            dataset_num_proc=2,
        )

        target_unsloth.verify_text_preprocessing(
            trainer,
            dataset_from_dict=mock.Mock(return_value="verification-dataset"),
            processing_class=tokenizer,
            policy=policy,
        )

        self.assertEqual(trainer.data_collator.call_count, 2)

    def test_rejects_masked_non_packed_start_label(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        trainer = self.non_packed_verification(
            tokenizer,
            policy,
            mask_starts=True,
        )
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            max_length=512,
            dataset_num_proc=2,
        )

        with self.assertRaisesRegex(RuntimeError, "full-sequence labels"):
            target_unsloth.verify_text_preprocessing(
                trainer,
                dataset_from_dict=mock.Mock(return_value="verification-dataset"),
                processing_class=tokenizer,
                policy=policy,
            )

    def test_allows_padding_free_document_start_mask(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        trainer = self.non_packed_verification(
            tokenizer,
            policy,
            mask_starts=True,
        )
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=True,
            max_length=512,
            dataset_num_proc=2,
        )

        target_unsloth.verify_text_preprocessing(
            trainer,
            dataset_from_dict=mock.Mock(return_value="verification-dataset"),
            processing_class=tokenizer,
            policy=policy,
        )

    def test_rejects_missing_packing_boundaries(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        _, _, prepared_record = self.packed_verification_record(tokenizer, policy)
        prepared_record.pop("seq_lengths")
        trainer = mock.Mock()
        trainer.args = SimpleNamespace(
            packing=True,
            max_length=512,
            dataset_num_proc=2,
        )
        trainer._prepare_dataset.return_value = [prepared_record]

        with self.assertRaisesRegex(
            RuntimeError,
            "packing did not preserve sequence lengths",
        ):
            target_unsloth.verify_text_preprocessing(
                trainer,
                dataset_from_dict=mock.Mock(return_value="verification-dataset"),
                processing_class=tokenizer,
                policy=policy,
            )

    def test_rejects_duplicate_effective_bos(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        _, _, prepared_record = self.packed_verification_record(tokenizer, policy)
        prepared_record["input_ids"].insert(0, tokenizer.bos_token_id)
        prepared_record["seq_lengths"][0] += 1
        trainer = mock.Mock()
        trainer.args = SimpleNamespace(
            packing=True,
            max_length=512,
            dataset_num_proc=2,
        )
        trainer._prepare_dataset.return_value = [prepared_record]

        with self.assertRaisesRegex(
            RuntimeError,
            "exactly one leading BOS per record",
        ):
            target_unsloth.verify_text_preprocessing(
                trainer,
                dataset_from_dict=mock.Mock(return_value="verification-dataset"),
                processing_class=tokenizer,
                policy=policy,
            )

    def test_rejects_masked_non_boundary_labels(self):
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        policy = target_unsloth.text_boundary_policy(tokenizer)
        _, _, prepared_record = self.packed_verification_record(tokenizer, policy)
        labels = list(prepared_record["input_ids"])
        labels[1] = -100
        trainer = mock.Mock()
        trainer.args = SimpleNamespace(
            packing=True,
            max_length=512,
            dataset_num_proc=2,
        )
        trainer._prepare_dataset.return_value = [prepared_record]
        trainer.data_collator.return_value = {
            "input_ids": [prepared_record["input_ids"]],
            "labels": [labels],
        }

        with self.assertRaisesRegex(RuntimeError, "full-sequence labels"):
            target_unsloth.verify_text_preprocessing(
                trainer,
                dataset_from_dict=mock.Mock(return_value="verification-dataset"),
                processing_class=tokenizer,
                policy=policy,
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
        train_config["config"]["unsloth"]["packing"] = True
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
        trainer.train_dataset = [verification_record, verification_record]
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
        self.assertIs(
            dependencies.sft_config.call_args.kwargs.get("packing"),
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

    def test_uses_effective_packing_when_unsloth_disables_packing(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "prompt-completion"
        train_config["config"]["unsloth"]["packing"] = True
        dataset = in_memory_dataset(
            [{"prompt": "Question?", "completion": " Answer."}]
        )
        dependencies = example_train_dependencies(dataset)
        trainer = dependencies.sft_trainer.return_value
        trainer.args.packing = False
        unpacked_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer._prepare_dataset.return_value = [unpacked_record]
        trainer.train_dataset = [unpacked_record]
        trainer.data_collator.return_value = {
            "input_ids": [unpacked_record["input_ids"]],
            "labels": [[-100, -100, 13, 2]],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        self.assertIs(
            dependencies.sft_config.call_args.kwargs.get("packing"),
            True,
        )
        self.assertNotIn("seq_lengths", unpacked_record)
        trainer.train.assert_called_once_with()

    def test_rejects_real_record_truncated_before_eos(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "prompt-completion"
        train_config["config"]["unsloth"]["maxSeqLength"] = 3
        dataset = in_memory_dataset(
            [{"prompt": "A prompt that fills the window", "completion": " Answer."}]
        )
        dependencies = example_train_dependencies(dataset)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                RuntimeError,
                r"record 0.*does not end with a supervised EOS token.*maxSeqLength 3",
            ):
                target_unsloth.train_model(
                    train_config,
                    trained_model_directory=Path(temporary_directory)
                    / "trained-model",
                    dependencies=dependencies,
                )

        dependencies.sft_trainer.assert_not_called()

    def test_rejects_packed_segment_truncated_before_eos(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "prompt-completion"
        train_config["config"]["unsloth"]["packing"] = True
        train_config["config"]["unsloth"]["maxSeqLength"] = 8
        dataset = in_memory_dataset(
            [
                {"prompt": "Question one?", "completion": " Answer one."},
                {"prompt": "Question two?", "completion": " Answer two."},
            ]
        )
        dependencies = example_train_dependencies(dataset)
        trainer = dependencies.sft_trainer.return_value
        trainer.args.packing = True
        verification_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer._prepare_dataset.return_value = [verification_record]
        trainer.data_collator.return_value = {
            "input_ids": [verification_record["input_ids"]],
            "labels": [[-100, -100, 13, 2]],
        }
        trainer.train_dataset = [
            {
                "input_ids": [11, 12, 13, 2, 11, 12, 21, 22],
                "completion_mask": [0, 0, 1, 1, 0, 0, 1, 1],
                "seq_lengths": [4, 4],
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                RuntimeError,
                r"record 0 packed segment 1.*supervised EOS.*maxSeqLength 8",
            ):
                target_unsloth.train_model(
                    train_config,
                    trained_model_directory=Path(temporary_directory)
                    / "trained-model",
                    dependencies=dependencies,
                )

        trainer.train.assert_not_called()

    def test_accepts_real_record_with_changed_tokenized_prompt_boundary(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "prompt-completion"
        dataset = in_memory_dataset(
            [{"prompt": "Boundary-sensitive prompt", "completion": " Answer."}]
        )
        dependencies = example_train_dependencies(dataset)
        tokenizer = dependencies.fast_language_model.from_pretrained.return_value[1]
        trainer = dependencies.sft_trainer.return_value
        verification_record = {
            "input_ids": [11, 12, 13, 2],
            "completion_mask": [0, 0, 1, 1],
        }
        trainer._prepare_dataset.return_value = [verification_record]
        trainer.train_dataset = [
            {
                "input_ids": [11, 99, 13, 2],
                "completion_mask": [0, 0, 1, 1],
            }
        ]
        trainer.data_collator.return_value = {
            "input_ids": [verification_record["input_ids"]],
            "labels": [[-100, -100, 13, 2]],
        }
        default_tokenize = tokenizer.side_effect

        def tokenize(text, *, add_special_tokens):
            if text == [
                "Boundary-sensitive prompt",
                "Boundary-sensitive prompt Answer.<eos>",
            ]:
                return {"input_ids": [[11, 12], [11, 99, 13, 2]]}
            return default_tokenize(
                text,
                add_special_tokens=add_special_tokens,
            )

        tokenizer.side_effect = tokenize

        with tempfile.TemporaryDirectory() as temporary_directory:
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        trainer.train.assert_called_once_with()

    def test_text_dataset_uses_isolated_normalization_and_full_sequence_loss(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "text"
        rows = [
            {
                "text": "<bos>Preformatted sequence.",
                "instruction": "must not use the Alpaca formatter",
                "output": "must not use the Alpaca formatter",
            },
            {
                "text": "Already terminated.<eos>",
                "prompt": "must not use prompt-completion dispatch",
                "completion": "must not use prompt-completion dispatch",
            },
        ]
        dataset = in_memory_dataset(rows)
        normalized_dataset = mock.Mock()
        dataset.projected_dataset.map.return_value = normalized_dataset
        dependencies = example_train_dependencies(dataset)
        fast_language_model = dependencies.fast_language_model
        base_model = fast_language_model.from_pretrained.return_value[0]
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        fast_language_model.from_pretrained.return_value = (base_model, tokenizer)
        trainer = dependencies.sft_trainer.return_value

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                target_unsloth,
                "verify_text_preprocessing",
            ) as verify_text_preprocessing,
        ):
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        dataset.select_columns.assert_called_once_with(["text"])
        dataset.map.assert_not_called()
        dataset.projected_dataset.map.assert_called_once_with(
            mock.ANY,
            batched=True,
        )
        normalize_batch = dataset.projected_dataset.map.call_args.args[0]
        self.assertEqual(
            normalize_batch({"text": [row["text"] for row in rows]}),
            {
                "text": [
                    "Preformatted sequence.<eos>",
                    "Already terminated.<eos>",
                ]
            },
        )
        dependencies.sft_trainer.assert_called_once_with(
            model=fast_language_model.get_peft_model.return_value,
            train_dataset=normalized_dataset,
            processing_class=tokenizer,
            args="sft-config",
        )
        self.assertIs(
            dependencies.sft_config.call_args.kwargs.get("completion_only_loss"),
            False,
        )
        verify_text_preprocessing.assert_called_once_with(
            trainer,
            dataset_from_dict=dependencies.dataset_from_dict,
            processing_class=tokenizer,
            policy=mock.ANY,
        )
        tokenizer.save_pretrained.assert_called_once()

    def test_text_dataset_requires_eos_before_lora_allocation(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "text"
        dataset = in_memory_dataset([{"text": "Preformatted sequence."}])
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = TextPreprocessingTokenizer(
            eos_token=None,
            eos_token_id=None,
        )
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "non-empty tokenizer EOS token",
        ):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.resolve_model_name.assert_not_called()
        dependencies.model_info.assert_not_called()
        dependencies.sft_trainer.assert_not_called()

    def test_text_dataset_rejects_overflow_before_lora_allocation(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "text"
        train_config["config"]["unsloth"]["maxSeqLength"] = 4
        dataset = in_memory_dataset([{"text": "ABC"}])
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = TextPreprocessingTokenizer(auto_bos=True)
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )

        with self.assertRaisesRegex(
            ValueError,
            "source 'organization/dataset' row 0.*at least 5 tokens.*maxSeqLength 4",
        ):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.resolve_model_name.assert_not_called()
        dependencies.model_info.assert_not_called()
        dataset.projected_dataset.map.assert_not_called()
        dependencies.sft_trainer.assert_not_called()

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
        valid_text = {"text": "A complete preformatted sequence."}
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
                "empty text dataset",
                "text",
                [],
                "text dataset source 'organization/dataset'.*at least one",
            ),
            (
                "missing text column",
                "text",
                [{"content": "Not a text field."}],
                "text dataset source 'organization/dataset'.*text",
            ),
            (
                "null text",
                "text",
                [valid_text, {"text": None}],
                "source 'organization/dataset' row 1.*text.*string",
            ),
            (
                "non-string text",
                "text",
                [valid_text, {"text": ["not", "a", "string"]}],
                "source 'organization/dataset' row 1.*text.*string",
            ),
            (
                "empty text value",
                "text",
                [valid_text, {"text": ""}],
                "source 'organization/dataset' row 1.*text.*non-empty string",
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
