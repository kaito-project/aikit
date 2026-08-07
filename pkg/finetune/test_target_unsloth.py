import hashlib
import importlib.util
import io
import json
import math
import tempfile
import threading
import unittest
from contextlib import nullcontext, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).with_name("target_unsloth.py")
MODULE_SPEC = importlib.util.spec_from_file_location("target_unsloth", MODULE_PATH)
target_unsloth = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(target_unsloth)




class LocalDatasetServer:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(self.path)
                path = self.path.split("?", 1)[0]
                body = outer.responses.get(path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *args):
                del args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def url(self, path):
        host, port = self.server.server_address
        return f"http://{host}:{port}{path}"


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


def example_dpo_train_config():
    train_config = example_train_config()
    train_config["objective"] = {
        "type": "dpo",
        "beta": 0.1,
        "lossType": "sigmoid",
        "maxPromptLength": 512,
    }
    train_config["datasets"] = [
        {"source": "organization/preferences", "type": "preference"}
    ]
    train_config["config"]["unsloth"]["learningRate"] = 0.000001
    return train_config


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


class FunctionalDataset:
    def __init__(self, rows):
        self.rows = tuple(dict(row) for row in rows)
        self.column_names = list(
            dict.fromkeys(key for row in self.rows for key in row)
        )
        self.select_columns = mock.Mock(side_effect=self._select_columns)
        self.map = mock.Mock(side_effect=self._map)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def _select_columns(self, column_names):
        projected_dataset = FunctionalDataset(
            {
                column_name: row[column_name]
                for column_name in column_names
                if column_name in row
            }
            for row in self.rows
        )
        self.projected_dataset = projected_dataset
        return projected_dataset

    def _map(
        self,
        function,
        *,
        batched=False,
        batch_size=None,
        with_indices=False,
        remove_columns=None,
        writer_batch_size=None,
    ):
        removed = set(remove_columns or ())
        mapped_rows = []
        if not batched:
            for row_index, row in enumerate(self.rows):
                if with_indices:
                    output = function(dict(row), row_index)
                else:
                    output = function(dict(row))
                mapped_row = {
                    key: value for key, value in row.items() if key not in removed
                }
                mapped_row.update(output)
                mapped_rows.append(mapped_row)
            return FunctionalDataset(mapped_rows)

        if batch_size is None:
            batch_size = len(self.rows) or 1
        for batch_start in range(0, len(self.rows), batch_size):
            rows = self.rows[batch_start : batch_start + batch_size]
            examples = {
                column_name: [row[column_name] for row in rows]
                for column_name in self.column_names
            }
            if with_indices:
                output = function(
                    examples,
                    list(range(batch_start, batch_start + len(rows))),
                )
            else:
                output = function(examples)
            for batch_index, row in enumerate(rows):
                mapped_row = {
                    key: value for key, value in row.items() if key not in removed
                }
                mapped_row.update(
                    {
                        key: values[batch_index]
                        for key, values in output.items()
                    }
                )
                mapped_rows.append(mapped_row)
        return FunctionalDataset(mapped_rows)


class MessagesTokenizer:
    def __init__(
        self,
        *,
        chat_template="{{ bos_token }}{{ messages }}{{ eos_token }}",
        bos_token="<bos>",
        bos_token_id=1,
        eos_token="<eos>",
        eos_token_id=2,
    ):
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.apply_calls = []
        self.text_calls = []
        self.render_error = None
        self.tokenize_error = None
        self.text_error = None
        self.mutate_rendered_tokenization = False
        self.save_pretrained = mock.Mock()

    def render(self, messages):
        prefix = self.bos_token or ""
        body = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return prefix + body + self.eos_token

    def raw_token_ids(self, text):
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
        return input_ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        truncation=False,
        max_length=None,
        return_dict=True,
    ):
        self.apply_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "truncation": truncation,
                "max_length": max_length,
                "return_dict": return_dict,
            }
        )
        if not tokenize and self.render_error is not None:
            raise self.render_error
        if tokenize and self.tokenize_error is not None:
            raise self.tokenize_error

        rendered = self.render(messages)
        if not tokenize:
            return rendered
        input_ids = self.raw_token_ids(rendered)
        if truncation:
            if max_length is None:
                raise AssertionError("truncation requires max_length")
            input_ids = input_ids[:max_length]
        if return_dict:
            return {"input_ids": input_ids}
        return input_ids

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        if self.text_error is not None:
            raise self.text_error
        self.text_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "max_length": max_length,
            }
        )

        def token_ids(value):
            input_ids = self.raw_token_ids(value)
            if add_special_tokens and self.bos_token_id is not None:
                input_ids.insert(0, self.bos_token_id)
            if self.mutate_rendered_tokenization and input_ids:
                input_ids[0] += 100
            if truncation:
                if max_length is None:
                    raise AssertionError("truncation requires max_length")
                input_ids = input_ids[:max_length]
            return input_ids

        if isinstance(text, list):
            return {"input_ids": [token_ids(value) for value in text]}
        return {"input_ids": token_ids(text)}


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
        dpo_config=mock.Mock(return_value="dpo-config"),
        dpo_trainer=mock.Mock(),
        get_chat_template_parts=mock.Mock(
            return_value=("<user>", "<assistant>")
        ),
        train_on_responses_only=mock.Mock(side_effect=lambda trainer, **_: trainer),
    )
    return dependencies


def example_dpo_train_dependencies(dataset, *, train_config=None):
    if train_config is None:
        train_config = example_dpo_train_config()
    dependencies = example_train_dependencies(dataset)
    tokenizer = dependencies.fast_language_model.from_pretrained.return_value[1]
    tokenizer.tokenizer = tokenizer
    trainer = mock.Mock()
    trainer.ref_model = None
    trainer.is_peft_model = True
    trainer.reference_free = False
    trainer.beta = train_config["objective"]["beta"]
    trainer.loss_type = [train_config["objective"]["lossType"]]
    trainer.max_prompt_length = train_config["objective"]["maxPromptLength"]
    trainer.max_length = train_config["config"]["unsloth"]["maxSeqLength"]
    trainer.model = dependencies.fast_language_model.get_peft_model.return_value
    trainer.accelerator = mock.Mock()
    trainer.accelerator.unwrap_model.side_effect = lambda model: model

    def construct_trainer(**kwargs):
        trainer.train_dataset = kwargs["train_dataset"]
        return trainer

    dependencies.dpo_config.return_value = "dpo-config"
    dependencies.dpo_trainer.side_effect = construct_trainer
    dependencies.dpo_trainer.runtime_trainer = trainer
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

    def test_sft_loss_defaults_to_all_and_limits_response_to_chat_data(self):
        train_config = example_train_config()
        self.assertEqual(
            target_unsloth.training_loss(
                train_config,
                dataset_type=target_unsloth.DATASET_TYPE_ALPACA,
            ),
            target_unsloth.LOSS_ALL,
        )

        train_config["config"]["unsloth"]["loss"] = None
        self.assertEqual(
            target_unsloth.training_loss(
                train_config,
                dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
            ),
            target_unsloth.LOSS_ALL,
        )

        train_config["config"]["unsloth"]["loss"] = "response"
        for dataset_type in (
            target_unsloth.DATASET_TYPE_MESSAGES,
            target_unsloth.DATASET_TYPE_SHAREGPT,
        ):
            with self.subTest(dataset_type=dataset_type):
                self.assertEqual(
                    target_unsloth.training_loss(
                        train_config,
                        dataset_type=dataset_type,
                    ),
                    target_unsloth.LOSS_RESPONSE,
                )

        for dataset_type in (
            target_unsloth.DATASET_TYPE_ALPACA,
            target_unsloth.DATASET_TYPE_PROMPT_COMPLETION,
            target_unsloth.DATASET_TYPE_TEXT,
        ):
            with self.subTest(dataset_type=dataset_type):
                with self.assertRaisesRegex(
                    ValueError,
                    "supported only for messages and sharegpt",
                ):
                    target_unsloth.training_loss(
                        train_config,
                        dataset_type=dataset_type,
                    )

        train_config["config"]["unsloth"]["packing"] = True
        train_config["config"]["unsloth"]["loss"] = "response"
        with self.assertRaisesRegex(ValueError, "does not support packing"):
            target_unsloth.training_loss(
                train_config,
                dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
            )

        train_config["config"]["unsloth"]["packing"] = False
        for invalid_loss in ("", "assistant"):
            train_config["config"]["unsloth"]["loss"] = invalid_loss
            with self.subTest(invalid_loss=invalid_loss):
                with self.assertRaisesRegex(ValueError, "unsupported SFT loss"):
                    target_unsloth.training_loss(
                        train_config,
                        dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
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


class TrainingObjectiveTest(unittest.TestCase):
    def test_defaults_to_sft_when_omitted_null_or_empty(self):
        for objective in (mock.sentinel.omitted, None, {}):
            with self.subTest(objective=objective):
                train_config = example_train_config()
                if objective is not mock.sentinel.omitted:
                    train_config["objective"] = objective
                self.assertEqual(
                    target_unsloth.training_objective_spec(train_config),
                    target_unsloth.TrainingObjectiveSpec(
                        objective_type="sft",
                        beta=None,
                        loss_type=None,
                        max_prompt_length=None,
                    ),
                )

    def test_parses_dpo_defaults_and_explicit_values(self):
        train_config = example_train_config()
        train_config["objective"] = {"type": "dpo"}
        self.assertEqual(
            target_unsloth.training_objective_spec(train_config),
            target_unsloth.TrainingObjectiveSpec(
                objective_type="dpo",
                beta=0.1,
                loss_type="sigmoid",
                max_prompt_length=512,
            ),
        )

        train_config["objective"] = {
            "type": "dpo",
            "beta": 0.25,
            "lossType": "sigmoid",
            "maxPromptLength": 128,
        }
        self.assertEqual(
            target_unsloth.training_objective_spec(train_config),
            target_unsloth.TrainingObjectiveSpec(
                objective_type="dpo",
                beta=0.25,
                loss_type="sigmoid",
                max_prompt_length=128,
            ),
        )

    def test_rejects_invalid_objective_shapes_and_fields(self):
        cases = (
            ([], "must be a mapping"),
            ({1: "dpo"}, "field names must be strings"),
            ({"type": "dpo", "future": True}, "unknown fields"),
            ({"type": ""}, "unsupported training objective"),
            ({"type": "orpo"}, "unsupported training objective"),
            ({"type": "sft", "beta": 0.1}, "does not support DPO fields"),
            ({"type": "sft", "lossType": None}, "does not support DPO fields"),
        )
        for objective, error_pattern in cases:
            with self.subTest(objective=objective):
                train_config = example_train_config()
                train_config["objective"] = objective
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.training_objective_spec(train_config)

    def test_rejects_invalid_dpo_values(self):
        cases = (
            ("beta", False, "beta.*finite"),
            ("beta", "0.1", "beta.*finite"),
            ("beta", 0, "beta.*finite"),
            ("beta", -0.1, "beta.*finite"),
            ("beta", math.nan, "beta.*finite"),
            ("beta", math.inf, "beta.*finite"),
            ("lossType", "", "unsupported DPO.*loss"),
            ("lossType", "hinge", "unsupported DPO.*loss"),
            ("maxPromptLength", False, "maxPromptLength.*integer"),
            ("maxPromptLength", 1.5, "maxPromptLength.*integer"),
            ("maxPromptLength", 0, "maxPromptLength.*integer"),
            ("maxPromptLength", -1, "maxPromptLength.*integer"),
        )
        for field, value, error_pattern in cases:
            with self.subTest(field=field, value=value):
                train_config = example_train_config()
                train_config["objective"] = {"type": "dpo", field: value}
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.training_objective_spec(train_config)

    def test_validates_objective_dataset_and_sft_settings(self):
        train_config = example_dpo_train_config()
        objective = target_unsloth.training_objective_spec(train_config)
        dataset_spec = target_unsloth.training_dataset_spec(train_config)
        self.assertEqual(
            target_unsloth.validate_training_objective(
                train_config,
                objective=objective,
                dataset_spec=dataset_spec,
            ),
            "all",
        )

        cases = (
            (
                "SFT preference",
                lambda config: config.pop("objective"),
                "preference datasets.*DPO",
            ),
            (
                "DPO Alpaca",
                lambda config: config["datasets"][0].update(type="alpaca"),
                "DPO objective requires a preference dataset",
            ),
            (
                "DPO packing",
                lambda config: config["config"]["unsloth"].update(
                    packing=True
                ),
                "DPO objective does not support packing",
            ),
            (
                "DPO non-boolean packing",
                lambda config: config["config"]["unsloth"].update(
                    packing="false"
                ),
                "packing must be a boolean",
            ),
            (
                "DPO response loss",
                lambda config: config["config"]["unsloth"].update(
                    loss="response"
                ),
                "response SFT loss.*DPO",
            ),
            (
                "DPO prompt too long",
                lambda config: config["objective"].update(
                    maxPromptLength=4096
                ),
                "maxPromptLength must not exceed",
            ),
            (
                "DPO text loader",
                lambda config: config["datasets"][0].update(
                    source="https://example.test/preferences.txt",
                    loader={"type": "text"},
                ),
                "do not support the text loader",
            ),
        )
        for name, mutate, error_pattern in cases:
            with self.subTest(name=name):
                invalid_config = example_dpo_train_config()
                mutate(invalid_config)
                invalid_objective = target_unsloth.training_objective_spec(
                    invalid_config
                )
                invalid_dataset = target_unsloth.training_dataset_spec(
                    invalid_config
                )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.validate_training_objective(
                        invalid_config,
                        objective=invalid_objective,
                        dataset_spec=invalid_dataset,
                    )


class DatasetSourceTest(unittest.TestCase):
    def legacy_spec(self, source):
        return target_unsloth.TrainingDatasetSpec(
            source=source,
            dataset_type=target_unsloth.DATASET_TYPE_TEXT,
            loader=target_unsloth.DatasetLoaderSpec(
                loader_type=None,
                subset=None,
                split="train",
                revision=None,
                checksum=None,
            ),
        )

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

    def test_builds_legacy_remote_json_load_spec_from_local_file(self):
        source = "https://example.test/train.json?query=opaque-marker"
        local_file = Path("/cache/content.json")

        self.assertEqual(
            target_unsloth.dataset_load_spec(
                self.legacy_spec(source),
                local_file=local_file,
            ),
            target_unsloth.DatasetLoadSpec(
                path="json",
                kwargs={
                    "data_files": {"train": str(local_file)},
                    "split": "train",
                },
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
            target_unsloth.dataset_load_spec(
                self.legacy_spec("organization/dataset")
            ),
            target_unsloth.DatasetLoadSpec(
                path="organization/dataset",
                kwargs={"split": "train"},
            ),
        )


class DatasetLoaderTest(unittest.TestCase):
    revision = "0123456789abcdef0123456789abcdef01234567"

    def training_spec(self, *, source, loader, dataset_type="text"):
        return target_unsloth.training_dataset_spec(
            {
                "datasets": [
                    {
                        "source": source,
                        "type": dataset_type,
                        "loader": loader,
                    }
                ]
            }
        )

    def test_loader_omission_preserves_legacy_defaults(self):
        spec = target_unsloth.training_dataset_spec(
            {
                "datasets": [
                    {
                        "source": "organization/dataset",
                        "type": "alpaca",
                    }
                ]
            }
        )

        self.assertEqual(
            spec.loader,
            target_unsloth.DatasetLoaderSpec(
                loader_type=None,
                subset=None,
                split="train",
                revision=None,
                checksum=None,
            ),
        )

    def test_legacy_url_still_uses_json_and_train_via_local_file(self):
        body = b'{"text":"legacy"}\n'
        with LocalDatasetServer({"/train.jsonl": body}) as server:
            spec = target_unsloth.training_dataset_spec(
                {
                    "datasets": [
                        {
                            "source": server.url("/train.jsonl"),
                            "type": "text",
                        }
                    ]
                }
            )
            observed = {}

            def load_dataset(_loader_type, *, data_files, split):
                local_path = Path(data_files[split])
                observed["path"] = local_path
                observed["bytes"] = local_path.read_bytes()
                return "dataset"

            load_dataset = mock.Mock(side_effect=load_dataset)
            with tempfile.TemporaryDirectory() as cache_directory:
                result = target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=load_dataset,
                    cache_directory=cache_directory,
                )
                self.assertNotEqual(
                    observed["path"].parent,
                    Path(cache_directory),
                )

        self.assertEqual(result, "dataset")
        self.assertEqual(observed["bytes"], body)
        self.assertFalse(observed["path"].exists())
        load_dataset.assert_called_once_with(
            "json",
            data_files={"train": str(observed["path"])},
            split="train",
        )

    def test_huggingface_loader_reaches_exact_datasets_api(self):
        spec = self.training_spec(
            source="HuggingFaceH4/ultrachat_200k",
            dataset_type="messages",
            loader={
                "type": "huggingface",
                "subset": "default",
                "split": "train_sft",
                "revision": self.revision,
            },
        )
        load_dataset = mock.Mock(return_value="dataset")

        result = target_unsloth.load_training_dataset(
            spec,
            load_dataset=load_dataset,
        )

        self.assertEqual(result, "dataset")
        load_dataset.assert_called_once_with(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            name="default",
            revision=self.revision,
        )

    def test_loader_validation_rejects_invalid_shapes_and_combinations(self):
        cases = (
            (None, "loader must be a mapping"),
            ([], "loader must be a mapping"),
            ({"type": "json", "splti": "train"}, "unknown fields"),
            ({"type": 1}, "field 'type' must be a string"),
            ({"type": ""}, "loader type must be a non-empty string"),
            ({"type": "arrow"}, "unsupported training dataset loader"),
            (
                {"type": "huggingface", "split": "train[:10%]"},
                "split must be a named split",
            ),
            (
                {"type": "huggingface", "split": "train-sft"},
                "split must be a named split",
            ),
            (
                {"type": "huggingface", "revision": "main"},
                "revision must be a lowercase 40-character commit hash",
            ),
            (
                {"type": "huggingface", "checksum": "sha256:" + "0" * 64},
                "does not support checksum",
            ),
            (
                {"type": "json", "subset": "default"},
                "do not support subset",
            ),
            (
                {"type": "json", "revision": self.revision},
                "do not support revision",
            ),
            (
                {"type": "json", "checksum": "sha256:0123"},
                "checksum must use lowercase",
            ),
        )
        for loader, error_pattern in cases:
            with self.subTest(loader=loader):
                source = (
                    "organization/dataset"
                    if loader and loader.get("type") == "huggingface"
                    else "https://example.test/train.json"
                )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    self.training_spec(source=source, loader=loader)

    def test_malformed_http_source_fails_without_leaking_details(self):
        source = (
            "https://user:password@[invalid-host/train.json?"
            "token=private#fragment"
        )
        with self.assertRaisesRegex(
            ValueError,
            "HTTP\\(S\\) source must be an absolute URL with a host",
        ) as raised:
            target_unsloth.training_dataset_spec(
                {"datasets": [{"source": source, "type": "text"}]}
            )

        for secret in (
            "user",
            "password",
            "invalid-host",
            "token",
            "private",
            "fragment",
        ):
            self.assertNotIn(secret, str(raised.exception))

    def test_loader_validation_rejects_transport_source_mismatches(self):
        with self.assertRaisesRegex(
            ValueError,
            "huggingface dataset loader does not support an HTTP",
        ):
            self.training_spec(
                source="https://example.test/train.json",
                loader={"type": "huggingface"},
            )
        with self.assertRaisesRegex(
            ValueError,
            "json dataset loader requires an absolute HTTP",
        ):
            self.training_spec(
                source="organization/dataset",
                loader={"type": "json"},
            )

    def test_remote_builders_download_verify_and_use_local_paths(self):
        cases = (
            ("json", "/train.jsonl", b'{"text":"json"}\n', ".jsonl"),
            ("csv", "/train.csv", b"text\ncsv\n", ".csv"),
            ("parquet", "/train.parquet", b"PAR1fixturePAR1", ".parquet"),
            ("text", "/train.txt", b"text fixture\n", ".txt"),
        )
        for loader_type, path, body, suffix in cases:
            with self.subTest(loader_type=loader_type):
                checksum = "sha256:" + hashlib.sha256(body).hexdigest()
                with LocalDatasetServer({path: body}) as server:
                    source = server.url(path) + "?token=opaque#fragment-marker"
                    spec = self.training_spec(
                        source=source,
                        loader={
                            "type": loader_type,
                            "split": "validation",
                            "checksum": checksum,
                        },
                    )
                    observed = {}

                    def parse_dataset(
                        _loader_type,
                        *,
                        data_files,
                        split,
                    ):
                        local_path = Path(data_files[split])
                        observed["path"] = local_path
                        observed["bytes"] = local_path.read_bytes()
                        observed["mode"] = local_path.stat().st_mode
                        return "dataset"

                    load_dataset = mock.Mock(side_effect=parse_dataset)
                    with tempfile.TemporaryDirectory() as cache_directory:
                        result = target_unsloth.load_training_dataset(
                            spec,
                            load_dataset=load_dataset,
                            cache_directory=cache_directory,
                        )
                        self.assertNotEqual(
                            observed["path"].parent,
                            Path(cache_directory),
                        )
                        cached_path = (
                            Path(cache_directory)
                            / f"{checksum.removeprefix('sha256:')}{suffix}"
                        )
                        self.assertEqual(cached_path.read_bytes(), body)

                local_path = observed["path"]
                self.assertEqual(result, "dataset")
                self.assertEqual(observed["bytes"], body)
                self.assertEqual(observed["mode"] & 0o777, 0o400)
                self.assertEqual(local_path.suffix, suffix)
                self.assertFalse(local_path.exists())
                self.assertNotIn("token", str(local_path))
                self.assertNotIn("fragment-marker", str(local_path))
                load_dataset.assert_called_once_with(
                    loader_type,
                    data_files={"validation": str(local_path)},
                    split="validation",
                )
                self.assertEqual(
                    server.requests,
                    [path + "?token=opaque"],
                )

    def test_checksum_mismatch_fails_before_parser(self):
        body = b'{"text":"changed"}\n'
        with LocalDatasetServer({"/train.json": body}) as server:
            source = server.url("/train.json")
            spec = self.training_spec(
                source=source,
                loader={
                    "type": "json",
                    "checksum": "sha256:" + "0" * 64,
                },
            )
            load_dataset = mock.Mock()
            with tempfile.TemporaryDirectory() as cache_directory:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "checksum does not match",
                ):
                    target_unsloth.load_training_dataset(
                        spec,
                        load_dataset=load_dataset,
                        cache_directory=cache_directory,
                    )
                cached_path = Path(cache_directory) / ("0" * 64 + ".json")
                self.assertFalse(cached_path.exists())
                self.assertTrue(
                    all(
                        path.name.startswith(".")
                        for path in Path(cache_directory).iterdir()
                    )
                )

        load_dataset.assert_not_called()

    def test_mutated_cached_bytes_are_reverified_before_parser(self):
        body = b"prompt,completion\nQuestion?,Answer.\n"
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        with LocalDatasetServer({"/train.csv": body}) as server:
            source = server.url("/train.csv")
            spec = self.training_spec(
                source=source,
                dataset_type="prompt-completion",
                loader={"type": "csv", "checksum": checksum},
            )
            observed_bytes = []
            observed_paths = []

            def parse_dataset(_loader_type, *, data_files, split):
                local_path = Path(data_files[split])
                observed_paths.append(local_path)
                observed_bytes.append(local_path.read_bytes())
                return "dataset"

            load_dataset = mock.Mock(side_effect=parse_dataset)
            with tempfile.TemporaryDirectory() as cache_directory:
                cached_path = (
                    Path(cache_directory)
                    / f"{checksum.removeprefix('sha256:')}.csv"
                )
                first = target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=load_dataset,
                    cache_directory=cache_directory,
                )
                cached_path.write_bytes(b"corrupt cached bytes")
                load_dataset.reset_mock()

                second = target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=load_dataset,
                    cache_directory=cache_directory,
                )

                self.assertEqual(cached_path.read_bytes(), body)
                load_dataset.assert_called_once_with(
                    "csv",
                    data_files={"train": str(observed_paths[-1])},
                    split="train",
                )

        self.assertEqual(first, "dataset")
        self.assertEqual(second, "dataset")
        self.assertEqual(observed_bytes, [body, body])
        self.assertTrue(all(not path.exists() for path in observed_paths))
        self.assertEqual(server.requests, ["/train.csv", "/train.csv"])

    def test_parser_uses_private_snapshot_after_cache_path_is_replaced(self):
        body = b'{"text":"verified"}\n'
        replacement = b'{"text":"unverified"}\n'
        digest = hashlib.sha256(body).hexdigest()
        checksum = f"sha256:{digest}"
        with LocalDatasetServer({"/train.json": body}) as server:
            spec = self.training_spec(
                source=server.url("/train.json"),
                loader={"type": "json", "checksum": checksum},
            )
            observed = {}
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                cache_directory = root / "cache"
                replacement_path = root / "replacement.json"
                replacement_path.write_bytes(replacement)
                cached_path = cache_directory / f"{digest}.json"

                def parse_dataset(_loader_type, *, data_files, split):
                    snapshot_path = Path(data_files[split])
                    observed["snapshot_path"] = snapshot_path
                    self.assertNotEqual(snapshot_path, cached_path)
                    cached_path.unlink()
                    cached_path.symlink_to(replacement_path)
                    observed["bytes"] = snapshot_path.read_bytes()
                    return "dataset"

                result = target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=parse_dataset,
                    cache_directory=cache_directory,
                )

                self.assertEqual(result, "dataset")
                self.assertEqual(observed["bytes"], body)
                self.assertTrue(cached_path.is_symlink())
                self.assertEqual(replacement_path.read_bytes(), replacement)
                self.assertFalse(observed["snapshot_path"].exists())

    def test_cache_entry_symlink_is_replaced_without_following_target(self):
        body = b'{"text":"verified"}\n'
        digest = hashlib.sha256(body).hexdigest()
        checksum = f"sha256:{digest}"
        with LocalDatasetServer({"/train.json": body}) as server:
            spec = self.training_spec(
                source=server.url("/train.json"),
                loader={"type": "json", "checksum": checksum},
            )
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                cache_directory = root / "cache"
                cache_directory.mkdir()
                symlink_target = root / "outside.json"
                symlink_target.write_bytes(body)
                cached_path = cache_directory / f"{digest}.json"
                cached_path.symlink_to(symlink_target)
                observed = []

                def parse_dataset(_loader_type, *, data_files, split):
                    observed.append(Path(data_files[split]).read_bytes())
                    return "dataset"

                result = target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=parse_dataset,
                    cache_directory=cache_directory,
                )

                self.assertEqual(result, "dataset")
                self.assertEqual(observed, [body])
                self.assertFalse(cached_path.is_symlink())
                self.assertEqual(cached_path.read_bytes(), body)
                self.assertEqual(symlink_target.read_bytes(), body)
                self.assertEqual(server.requests, ["/train.json"])

    def test_cache_lock_symlink_is_rejected_without_following_target(self):
        body = b'{"text":"verified"}\n'
        digest = hashlib.sha256(body).hexdigest()
        checksum = f"sha256:{digest}"
        with LocalDatasetServer({"/train.json": body}) as server:
            source = server.url("/train.json") + "?token=private#fragment"
            spec = self.training_spec(
                source=source,
                loader={"type": "json", "checksum": checksum},
            )
            parser = mock.Mock()
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                cache_directory = root / "cache"
                cache_directory.mkdir()
                lock_target = root / "outside.lock"
                lock_target.write_text("outside", encoding="utf-8")
                lock_path = cache_directory / f".{digest}.lock"
                lock_path.symlink_to(lock_target)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "cache lock could not be acquired safely",
                ) as raised:
                    target_unsloth.load_training_dataset(
                        spec,
                        load_dataset=parser,
                        cache_directory=cache_directory,
                    )

                self.assertEqual(lock_target.read_text(encoding="utf-8"), "outside")
                self.assertTrue(lock_path.is_symlink())
                self.assertEqual(server.requests, [])

        parser.assert_not_called()
        self.assertNotIn("private", str(raised.exception))
        self.assertNotIn("fragment", str(raised.exception))

    def test_per_digest_lock_serializes_corrupt_entry_replacement(self):
        body = b'{"text":"verified"}\n'
        digest = hashlib.sha256(body).hexdigest()
        checksum = f"sha256:{digest}"
        cache_started = threading.Event()
        release_download = threading.Event()
        second_download = threading.Event()
        call_lock = threading.Lock()
        open_calls = 0

        def open_url(_request_url):
            nonlocal open_calls
            with call_lock:
                open_calls += 1
                call_number = open_calls
            if call_number == 1:
                cache_started.set()
            else:
                second_download.set()
            if not release_download.wait(timeout=5):
                raise RuntimeError("test download was not released")
            return io.BytesIO(body)

        with tempfile.TemporaryDirectory() as cache_directory_name:
            cache_directory = Path(cache_directory_name)
            cached_path = cache_directory / f"{digest}.json"
            cached_path.write_bytes(b"corrupt")
            results = []
            errors = []
            second_started = threading.Event()

            def materialize(*, second=False):
                if second:
                    second_started.set()
                try:
                    with target_unsloth.materialize_remote_dataset_file(
                        "https://example.test/train.json",
                        loader_type="json",
                        checksum=checksum,
                        cache_directory=cache_directory,
                        open_url=open_url,
                    ) as snapshot_path:
                        results.append(snapshot_path.read_bytes())
                except Exception as error:
                    errors.append(error)

            first_thread = threading.Thread(target=materialize)
            second_thread = threading.Thread(
                target=materialize,
                kwargs={"second": True},
            )
            first_thread.start()
            self.assertTrue(cache_started.wait(timeout=2))
            second_thread.start()
            self.assertTrue(second_started.wait(timeout=2))
            try:
                self.assertFalse(second_download.wait(timeout=0.5))
            finally:
                release_download.set()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, [body, body])
            self.assertEqual(open_calls, 1)
            self.assertEqual(cached_path.read_bytes(), body)

    def test_unchecksummed_download_uses_actual_content_digest(self):
        body = b"line one\nline two\n"
        digest = hashlib.sha256(body).hexdigest()
        with LocalDatasetServer({"/download": body}) as server:
            with tempfile.TemporaryDirectory() as cache_directory:
                with target_unsloth.materialize_remote_dataset_file(
                    server.url("/download"),
                    loader_type="text",
                    checksum=None,
                    cache_directory=cache_directory,
                ) as snapshot_path:
                    self.assertEqual(snapshot_path.name, "dataset.txt")
                    self.assertEqual(snapshot_path.read_bytes(), body)

                cached_path = Path(cache_directory) / f"{digest}.txt"
                self.assertEqual(cached_path.read_bytes(), body)
                self.assertFalse(snapshot_path.exists())

    def test_default_cache_directory_uses_aikit_namespace(self):
        with tempfile.TemporaryDirectory() as cache_root:
            with mock.patch.dict(
                target_unsloth.os.environ,
                {"HF_DATASETS_CACHE": cache_root},
            ):
                self.assertEqual(
                    target_unsloth.dataset_cache_directory(),
                    Path(cache_root) / "aikit-remote-files",
                )

    def test_corrupt_cached_file_and_changed_download_never_reach_parser(self):
        original = b'{"text":"original"}\n'
        changed = b'{"text":"changed"}\n'
        checksum = "sha256:" + hashlib.sha256(original).hexdigest()
        with LocalDatasetServer({"/train.json": original}) as server:
            source = server.url("/train.json")
            spec = self.training_spec(
                source=source,
                loader={"type": "json", "checksum": checksum},
            )
            with tempfile.TemporaryDirectory() as cache_directory:
                initial_loader = mock.Mock(return_value="dataset")
                target_unsloth.load_training_dataset(
                    spec,
                    load_dataset=initial_loader,
                    cache_directory=cache_directory,
                )
                cached_path = (
                    Path(cache_directory)
                    / f"{checksum.removeprefix('sha256:')}.json"
                )
                cached_path.write_bytes(b"corrupt")
                server.responses["/train.json"] = changed
                parser = mock.Mock()

                with self.assertRaisesRegex(
                    RuntimeError,
                    "checksum does not match",
                ):
                    target_unsloth.load_training_dataset(
                        spec,
                        load_dataset=parser,
                        cache_directory=cache_directory,
                    )

                parser.assert_not_called()
                self.assertEqual(cached_path.read_bytes(), b"corrupt")

    def test_preserves_data_and_compression_suffixes(self):
        self.assertEqual(
            target_unsloth.remote_dataset_file_suffix(
                "https://example.test/train.jsonl.gz?token=value",
                "json",
            ),
            ".jsonl.gz",
        )
        self.assertEqual(
            target_unsloth.remote_dataset_file_suffix(
                "https://example.test/download.gz",
                "parquet",
            ),
            ".parquet.gz",
        )

    def test_download_exception_redacts_url_and_suppresses_cause(self):
        source = (
            "https://user:password@example.test/train.json?"
            "token=secret#fragment-marker"
        )

        def fail_with_url(_url):
            raise RuntimeError(f"failed to download {source}")

        with tempfile.TemporaryDirectory() as cache_directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "remote json dataset could not be downloaded",
            ) as raised:
                with target_unsloth.materialize_remote_dataset_file(
                    source,
                    loader_type="json",
                    checksum=None,
                    cache_directory=cache_directory,
                    open_url=fail_with_url,
                ):
                    self.fail("download failure should prevent materialization")

        message = str(raised.exception)
        for secret in (
            "user",
            "password",
            "example.test",
            "token",
            "secret",
            "fragment-marker",
        ):
            self.assertNotIn(secret, message)
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)


class MessagesSchemaTest(unittest.TestCase):
    def test_accepts_single_and_multi_turn_text_conversations(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "First question?"},
                    {"role": "assistant", "content": "First answer."},
                    {"role": "user", "content": "Follow-up?"},
                    {"role": "assistant", "content": ""},
                ]
            },
        ]

        target_unsloth.validate_training_dataset(
            rows,
            dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
            source="organization/chat-data",
        )

    def test_rejects_malformed_or_unsupported_conversations(self):
        valid_user = {"role": "user", "content": "Question?"}
        valid_assistant = {"role": "assistant", "content": "Answer."}
        cases = (
            ("empty list", [], "non-empty list"),
            (
                "non-list sequence",
                (valid_user, valid_assistant),
                "non-empty list",
            ),
            ("non-mapping message", [valid_user, "answer"], "message 1.*mapping"),
            (
                "missing role",
                [{"content": "Question?"}, valid_assistant],
                "message 0.*role",
            ),
            (
                "missing content",
                [{"role": "user"}, valid_assistant],
                "message 0.*content",
            ),
            (
                "metadata field",
                [
                    {"role": "user", "content": "Question?", "name": "alice"},
                    valid_assistant,
                ],
                "message 0.*unsupported fields.*name",
            ),
            (
                "nested id field",
                [
                    {
                        "role": "user",
                        "content": "Question?",
                        "id": "message-id",
                    },
                    valid_assistant,
                ],
                "message 0.*unsupported fields.*id",
            ),
            (
                "tool call field",
                [
                    valid_user,
                    {
                        "role": "assistant",
                        "content": "Answer.",
                        "tool_calls": [],
                    },
                ],
                "message 1.*unsupported fields.*tool_calls",
            ),
            (
                "non-string role",
                [{"role": 7, "content": "Question?"}, valid_assistant],
                "message 0.*role.*string",
            ),
            (
                "structured content",
                [
                    {"role": "user", "content": [{"type": "text"}]},
                    valid_assistant,
                ],
                "message 0.*content.*string",
            ),
            (
                "tool role",
                [valid_user, {"role": "tool", "content": "result"}],
                "unsupported role 'tool'",
            ),
            (
                "developer role",
                [
                    {"role": "developer", "content": "instructions"},
                    valid_assistant,
                ],
                "unsupported role 'developer'",
            ),
            (
                "no assistant",
                [valid_user],
                "at least one assistant",
            ),
            (
                "user final",
                [valid_user, valid_assistant, valid_user],
                "final message.*assistant",
            ),
        )

        for name, messages, error_pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.validate_training_dataset(
                        [{"messages": messages}],
                        dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
                        source="organization/chat-data",
                    )

    def test_accepts_benign_top_level_metadata_and_projects_to_messages(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        rows = [
            {
                "messages": messages,
                "id": "record-id",
                "source": "curated",
                "source_label": "support",
                "image_id": "benign-metadata-id",
                "quality_score": 0.98,
                "metadata": {"split": "train"},
            }
        ]

        target_unsloth.validate_training_dataset(
            rows,
            dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
            source="organization/chat-data",
        )
        dataset = FunctionalDataset(rows)
        projected_dataset = target_unsloth.project_training_dataset(
            dataset,
            dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
            source="organization/chat-data",
        )

        dataset.select_columns.assert_called_once_with(["messages"])
        self.assertEqual(projected_dataset.column_names, ["messages"])
        self.assertEqual(list(projected_dataset), [{"messages": messages}])

    def test_rejects_semantic_top_level_fields(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        for field, value in (
            ("add_generation_prompt", False),
            ("continue_final_message", False),
            ("tools", []),
            ("tool_choice", "auto"),
            ("function_call", {}),
            ("functions", []),
            ("documents", []),
            ("chat_template", "{{ messages }}"),
            ("chat_template_kwargs", {}),
            ("tokenizer_kwargs", {}),
            ("image", object()),
            ("images", []),
            ("image_path", "/dataset/image.png"),
            ("image_url", "https://example.test/image.png"),
            ("audio", object()),
            ("audios", []),
            ("audio_path", "/dataset/audio.wav"),
            ("audio_url", "https://example.test/audio.wav"),
            ("video", object()),
            ("videos", []),
            ("video_path", "/dataset/video.mp4"),
            ("video_url", "https://example.test/video.mp4"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"unsupported top-level fields.*{field}",
                ):
                    target_unsloth.validate_training_dataset(
                        [{"messages": messages, field: value}],
                        dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
                        source="organization/chat-data",
                    )

    def test_rejects_semantic_columns_before_projection(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = in_memory_dataset(
            [{"messages": messages, "id": "record-id", "tools": []}]
        )

        with self.assertRaisesRegex(
            ValueError,
            r"unsupported top-level fields.*tools",
        ):
            target_unsloth.project_training_dataset(
                dataset,
                dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
                source="organization/chat-data",
            )

        dataset.select_columns.assert_not_called()

    def test_redacts_signed_url_in_schema_errors(self):
        source = (
            "https://alice:password@example.test/private/train.json?"
            "X-Amz-Credential=credential-marker&X-Amz-Signature=signature-marker"
            "#fragment-marker"
        )

        with self.assertRaises(ValueError) as raised:
            target_unsloth.validate_training_dataset(
                [{"messages": []}],
                dataset_type=target_unsloth.DATASET_TYPE_MESSAGES,
                source=source,
            )

        message = str(raised.exception)
        self.assertIn("messages dataset remote JSON URL row 0", message)
        for secret in (
            source,
            "alice",
            "password",
            "credential-marker",
            "signature-marker",
            "fragment-marker",
        ):
            self.assertNotIn(secret, message)


class ShareGPTAdapterTest(unittest.TestCase):
    def test_normalizes_one_row_with_exact_role_map_and_bounded_mapping(self):
        conversations = [
            {"from": "system", "value": "Be concise."},
            {"from": "human", "value": "First question?"},
            {"from": "gpt", "value": "First answer."},
            {"from": "user", "value": "Follow-up?"},
            {"from": "assistant", "value": "Follow-up answer."},
        ]
        dataset = FunctionalDataset(
            [
                {
                    "conversations": conversations,
                    "id": "one-row-regression",
                    "metadata": {"source": "curated"},
                }
            ]
        )

        projected = target_unsloth.project_training_dataset(
            dataset,
            dataset_type=target_unsloth.DATASET_TYPE_SHAREGPT,
            source="organization/sharegpt-data",
        )
        target_unsloth.validate_training_dataset(
            projected,
            dataset_type=target_unsloth.DATASET_TYPE_SHAREGPT,
            source="organization/sharegpt-data",
        )
        normalized = target_unsloth.normalize_sharegpt_dataset(
            projected,
            source="organization/sharegpt-data",
        )

        self.assertEqual(
            list(normalized),
            [
                {
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "First question?"},
                        {"role": "assistant", "content": "First answer."},
                        {"role": "user", "content": "Follow-up?"},
                        {
                            "role": "assistant",
                            "content": "Follow-up answer.",
                        },
                    ]
                }
            ],
        )
        dataset.select_columns.assert_called_once_with(["conversations"])
        projected.map.assert_called_once_with(
            mock.ANY,
            batched=False,
            with_indices=True,
            remove_columns=["conversations"],
            writer_batch_size=1,
        )

    def test_rejects_malformed_conversations_without_heuristic_inference(self):
        valid_user = {"from": "human", "value": "Question?"}
        valid_assistant = {"from": "gpt", "value": "Answer."}
        cases = (
            ("empty", [], "conversations.*non-empty list"),
            (
                "non-list",
                (valid_user, valid_assistant),
                "conversations.*non-empty list",
            ),
            (
                "non-mapping",
                [valid_user, "answer"],
                "conversation 1.*mapping",
            ),
            (
                "missing from",
                [{"value": "Question?"}, valid_assistant],
                "conversation 0.*from",
            ),
            (
                "missing value",
                [{"from": "human"}, valid_assistant],
                "conversation 0.*value",
            ),
            (
                "heuristic role key",
                [{"role": "human", "value": "Question?"}, valid_assistant],
                "conversation 0.*from",
            ),
            (
                "heuristic content key",
                [{"from": "human", "content": "Question?"}, valid_assistant],
                "conversation 0.*value",
            ),
            (
                "unknown role",
                [{"from": "tool", "value": "Question?"}, valid_assistant],
                "unsupported role 'tool'",
            ),
            (
                "non-string role",
                [{"from": 7, "value": "Question?"}, valid_assistant],
                "from.*string",
            ),
            (
                "non-string content",
                [valid_user, {"from": "gpt", "value": ["Answer."]}],
                "value.*string",
            ),
            (
                "conversation metadata",
                [
                    {"from": "human", "value": "Question?", "name": "alice"},
                    valid_assistant,
                ],
                "unsupported fields.*name",
            ),
            ("no assistant", [valid_user], "at least one assistant"),
            (
                "user final",
                [valid_user, valid_assistant, valid_user],
                "final message.*assistant",
            ),
        )

        for name, conversations, error_pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.validate_training_dataset(
                        [{"conversations": conversations}],
                        dataset_type=target_unsloth.DATASET_TYPE_SHAREGPT,
                        source="organization/sharegpt-data",
                    )

    def test_rejects_semantic_top_level_fields_and_redacts_url_errors(self):
        conversations = [
            {"from": "human", "value": "Question?"},
            {"from": "gpt", "value": "Answer."},
        ]
        dataset = in_memory_dataset(
            [{"conversations": conversations, "tools": []}]
        )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported top-level fields.*tools",
        ):
            target_unsloth.project_training_dataset(
                dataset,
                dataset_type=target_unsloth.DATASET_TYPE_SHAREGPT,
                source="organization/sharegpt-data",
            )

        source = (
            "https://alice:password@example.test/private/train.json?"
            "X-Amz-Credential=credential-marker&X-Amz-Signature=signature-marker"
            "#fragment-marker"
        )
        with self.assertRaises(ValueError) as raised:
            target_unsloth.validate_training_dataset(
                [{"conversations": []}],
                dataset_type=target_unsloth.DATASET_TYPE_SHAREGPT,
                source=source,
            )

        message = str(raised.exception)
        self.assertIn("sharegpt dataset remote JSON URL row 0", message)
        for secret in (
            source,
            "alice",
            "password",
            "credential-marker",
            "signature-marker",
            "fragment-marker",
        ):
            self.assertNotIn(secret, message)


class MessagesRenderingTest(unittest.TestCase):
    def messages_rows(self):
        duplicate = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        return [
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    *duplicate,
                    {"role": "user", "content": "Follow-up?"},
                    {"role": "assistant", "content": "Follow-up answer."},
                ]
            },
            {"messages": duplicate},
            {"messages": duplicate},
        ]

    def render_dataset(self, rows, tokenizer):
        source_dataset = FunctionalDataset(rows)
        rendered_dataset = target_unsloth.render_messages_dataset(
            source_dataset,
            processing_class=tokenizer,
            source="organization/chat-data",
        )
        return source_dataset, rendered_dataset

    def test_requires_a_usable_model_chat_template(self):
        cases = (
            (MessagesTokenizer(chat_template=None), "usable tokenizer chat template"),
            (MessagesTokenizer(chat_template="  "), "usable tokenizer chat template"),
            (
                MessagesTokenizer(chat_template={"tool_use": "template"}),
                "usable tokenizer chat template",
            ),
            (SimpleNamespace(chat_template="template"), "apply_chat_template"),
        )

        for tokenizer, error_pattern in cases:
            with self.subTest(error_pattern=error_pattern):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.require_messages_chat_template(tokenizer)

        target_unsloth.require_messages_chat_template(MessagesTokenizer())
        target_unsloth.require_messages_chat_template(
            MessagesTokenizer(
                chat_template={"default": "{{ messages }}"},
                bos_token=None,
                bos_token_id=None,
            )
        )

    def test_rejects_wall_clock_dependent_chat_templates(self):
        wall_clock_template = "{{ strftime_now('%Y-%m-%d') }}{{ messages }}"
        inner_tokenizer = MessagesTokenizer(chat_template=wall_clock_template)
        cases = (
            MessagesTokenizer(chat_template=wall_clock_template),
            MessagesTokenizer(
                chat_template={"default": wall_clock_template},
                bos_token=None,
                bos_token_id=None,
            ),
            SimpleNamespace(
                tokenizer=inner_tokenizer,
                chat_template="",
                apply_chat_template=inner_tokenizer.apply_chat_template,
            ),
        )

        for tokenizer in cases:
            with self.subTest(tokenizer=tokenizer):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"wall-clock-dependent.*strftime_now.*cache keys",
                ):
                    target_unsloth.require_messages_chat_template(tokenizer)

    def test_renders_one_conversation_at_a_time_with_bounded_writer_buffer(self):
        tokenizer = MessagesTokenizer()
        rows = self.messages_rows()
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)

        self.assertEqual(rendered_dataset.column_names, ["text"])
        self.assertEqual(
            [row["text"] for row in rendered_dataset],
            [tokenizer.render(row["messages"]) for row in rows],
        )
        source_dataset.map.assert_called_once_with(
            mock.ANY,
            batched=False,
            with_indices=True,
            remove_columns=["messages"],
            writer_batch_size=1,
        )
        self.assertEqual(len(tokenizer.apply_calls), len(rows))
        for call in tokenizer.apply_calls:
            self.assertFalse(call["tokenize"])
            self.assertFalse(call["add_generation_prompt"])

    def test_render_failure_redacts_source_and_suppresses_sensitive_cause(self):
        source = (
            "https://alice:password@example.test/train.json?"
            "token=credential-marker#fragment-marker"
        )
        tokenizer = MessagesTokenizer()
        tokenizer.render_error = RuntimeError(source)

        with self.assertRaises(RuntimeError) as raised:
            target_unsloth.render_messages_example(
                {"messages": self.messages_rows()[0]["messages"]},
                7,
                processing_class=tokenizer,
                source_description=target_unsloth.dataset_source_description(
                    source
                ),
            )

        self.assertIsNone(raised.exception.__cause__)
        message = str(raised.exception)
        self.assertIn("messages dataset remote JSON URL row 7", message)
        for secret in (source, "alice", "password", "credential-marker"):
            self.assertNotIn(secret, message)

    def test_dataset_map_failure_redacts_source_and_suppresses_cause(self):
        source = (
            "https://alice:password@example.test/train.json?"
            "token=credential-marker#fragment-marker"
        )
        dataset = mock.Mock(column_names=["messages"])
        dataset.map.side_effect = RuntimeError(source)

        with self.assertRaisesRegex(
            RuntimeError,
            "messages dataset remote JSON URL could not be rendered",
        ) as raised:
            target_unsloth.render_messages_dataset(
                dataset,
                processing_class=MessagesTokenizer(),
                source=source,
            )

        message = str(raised.exception)
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)
        for secret in (source, "alice", "password", "credential-marker"):
            self.assertNotIn(secret, message)

    def test_matches_canonical_ids_with_bounded_locked_text_tokenization(self):
        tokenizer = MessagesTokenizer()
        rows = self.messages_rows()
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)
        tokenizer.apply_calls.clear()

        fingerprint = target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=tokenizer,
            max_seq_length=512,
            source="organization/chat-data",
            batch_size=2,
        )

        self.assertEqual(fingerprint.sequence_count, len(rows))
        self.assertEqual(
            [len(call["text"]) for call in tokenizer.text_calls],
            [2, 1],
        )
        for call in tokenizer.text_calls:
            self.assertFalse(call["add_special_tokens"])
            self.assertTrue(call["truncation"])
            self.assertEqual(call["max_length"], 513)
        self.assertEqual(len(tokenizer.apply_calls), len(rows))
        for call in tokenizer.apply_calls:
            self.assertTrue(call["tokenize"])
            self.assertFalse(call["add_generation_prompt"])
            self.assertTrue(call["truncation"])
            self.assertEqual(call["max_length"], 513)
            self.assertFalse(call["return_dict"])

    def test_response_markers_match_roles_and_reject_content_collisions(self):
        tokenizer = MessagesTokenizer()
        markers = target_unsloth.derive_response_markers(
            tokenizer,
            get_chat_template_parts=mock.Mock(
                return_value=("<user>", "<assistant>")
            ),
        )
        rows = [self.messages_rows()[1]]
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)

        fingerprint = target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=tokenizer,
            max_seq_length=512,
            source="organization/chat-data",
            response_markers=markers,
        )

        self.assertEqual(fingerprint.sequence_count, 1)

        collision_rows = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Quote <assistant> in the prompt.",
                    },
                    {"role": "assistant", "content": "Answer."},
                ]
            }
        ]
        source_dataset, rendered_dataset = self.render_dataset(
            collision_rows,
            tokenizer,
        )
        with self.assertRaisesRegex(ValueError, "content collides"):
            target_unsloth.validate_messages_tokenization(
                source_dataset,
                rendered_dataset,
                processing_class=tokenizer,
                max_seq_length=512,
                source="organization/chat-data",
                response_markers=markers,
            )

    def test_matches_qwen_like_tokenizer_without_bos(self):
        tokenizer = MessagesTokenizer(
            chat_template="{{ messages }}{{ eos_token }}",
            bos_token=None,
            bos_token_id=None,
        )
        rows = [self.messages_rows()[1]]
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)

        fingerprint = target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=tokenizer,
            max_seq_length=512,
            source="organization/chat-data",
        )

        self.assertEqual(fingerprint.sequence_count, 1)
        self.assertEqual(len(tokenizer.text_calls), 1)
        self.assertTrue(tokenizer.text_calls[0]["add_special_tokens"])

    def test_matches_locked_unsloth_tokenizer_wrapper_selection(self):
        inner_tokenizer = MessagesTokenizer()
        processing_class = SimpleNamespace(
            tokenizer=inner_tokenizer,
            chat_template="",
            bos_token=None,
            apply_chat_template=inner_tokenizer.apply_chat_template,
        )
        processing_class.__call__ = mock.Mock(
            side_effect=AssertionError("outer processor must not tokenize text")
        )
        rows = [self.messages_rows()[1]]
        source_dataset, rendered_dataset = self.render_dataset(
            rows,
            processing_class,
        )

        target_unsloth.require_messages_chat_template(processing_class)
        fingerprint = target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=processing_class,
            max_seq_length=512,
            source="organization/chat-data",
        )

        self.assertEqual(fingerprint.sequence_count, 1)
        self.assertEqual(len(inner_tokenizer.text_calls), 1)
        self.assertFalse(inner_tokenizer.text_calls[0]["add_special_tokens"])
        processing_class.__call__.assert_not_called()

    def test_accepts_exact_limit_and_rejects_overflow_before_truncation(self):
        tokenizer = MessagesTokenizer()
        rows = [self.messages_rows()[1]]
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)
        token_count = len(tokenizer.raw_token_ids(rendered_dataset[0]["text"]))

        target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=tokenizer,
            max_seq_length=token_count,
            source="organization/chat-data",
        )
        with self.assertRaisesRegex(
            ValueError,
            rf"row 0.*at least {token_count} tokens.*maxSeqLength {token_count - 1}",
        ):
            target_unsloth.validate_messages_tokenization(
                source_dataset,
                rendered_dataset,
                processing_class=tokenizer,
                max_seq_length=token_count - 1,
                source="organization/chat-data",
            )

    def test_rejects_canonical_and_locked_text_token_mismatch(self):
        tokenizer = MessagesTokenizer()
        rows = [self.messages_rows()[1]]
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)
        tokenizer.mutate_rendered_tokenization = True

        with self.assertRaisesRegex(
            RuntimeError,
            "canonical chat-template token IDs do not match.*locked Unsloth",
        ):
            target_unsloth.validate_messages_tokenization(
                source_dataset,
                rendered_dataset,
                processing_class=tokenizer,
                max_seq_length=512,
                source="organization/chat-data",
            )

    def test_tokenization_errors_redact_source_and_suppress_causes(self):
        source = (
            "https://alice:password@example.test/train.json?"
            "token=credential-marker#fragment-marker"
        )
        rows = [self.messages_rows()[1]]
        cases = ("tokenize_error", "text_error")

        for error_attribute in cases:
            with self.subTest(error_attribute=error_attribute):
                tokenizer = MessagesTokenizer()
                setattr(tokenizer, error_attribute, RuntimeError(source))
                source_dataset, rendered_dataset = self.render_dataset(
                    rows,
                    tokenizer,
                )

                with self.assertRaises(RuntimeError) as raised:
                    target_unsloth.validate_messages_tokenization(
                        source_dataset,
                        rendered_dataset,
                        processing_class=tokenizer,
                        max_seq_length=512,
                        source=source,
                    )

                message = str(raised.exception)
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)
                for secret in (
                    source,
                    "alice",
                    "password",
                    "credential-marker",
                ):
                    self.assertNotIn(secret, message)

    def test_limits_batches_by_retained_token_budget(self):
        tokenizer = MessagesTokenizer()
        row = self.messages_rows()[1]
        rows = [row for _ in range(65)]
        source_dataset, rendered_dataset = self.render_dataset(rows, tokenizer)
        tokenizer.text_calls.clear()

        target_unsloth.validate_messages_tokenization(
            source_dataset,
            rendered_dataset,
            processing_class=tokenizer,
            max_seq_length=4095,
            source="organization/chat-data",
        )

        self.assertEqual(
            [len(call["text"]) for call in tokenizer.text_calls],
            [32, 32, 1],
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


class MessagesPreprocessingContractTest(unittest.TestCase):
    def fingerprint(self, segments):
        fingerprint = target_unsloth.empty_messages_token_fingerprint()
        for index, segment in enumerate(segments):
            fingerprint = target_unsloth.extend_messages_token_fingerprint(
                fingerprint,
                segment,
                description=f"test segment {index}",
            )
        return fingerprint

    def packed_collator(self, records):
        record = records[0]
        input_ids = list(record["input_ids"])
        labels = list(input_ids)
        offset = 0
        for length in record["seq_lengths"]:
            labels[offset] = -100
            offset += length
        return {"input_ids": [input_ids], "labels": [labels]}

    def test_accepts_reordered_duplicate_bfd_segments_and_full_labels(self):
        first = [11, 12, 13]
        duplicate = [21, 22]
        final = [31, 32, 33]
        expected = self.fingerprint([first, duplicate, duplicate, final])
        prepared_dataset = [
            {
                "input_ids": final + duplicate,
                "seq_lengths": [len(final), len(duplicate)],
            },
            {
                "input_ids": duplicate + first,
                "seq_lengths": [len(duplicate), len(first)],
            },
        ]

        target_unsloth.validate_prepared_messages_dataset(
            prepared_dataset,
            data_collator=self.packed_collator,
            max_seq_length=16,
            packing=True,
            padding_free=True,
            packing_strategy="bfd",
            expected_fingerprint=expected,
        )

    def test_accepts_non_packed_and_padding_free_start_labels(self):
        segments = ([11, 12, 13], [21, 22])
        expected = self.fingerprint(segments)
        prepared_dataset = [{"input_ids": segment} for segment in segments]

        def full_collator(records):
            input_ids = list(records[0]["input_ids"])
            return {"input_ids": [input_ids], "labels": [list(input_ids)]}

        target_unsloth.validate_prepared_messages_dataset(
            prepared_dataset,
            data_collator=full_collator,
            max_seq_length=8,
            packing=False,
            padding_free=False,
            expected_fingerprint=expected,
        )

        def padding_free_collator(records):
            input_ids = list(records[0]["input_ids"])
            labels = list(input_ids)
            labels[0] = -100
            return {"input_ids": [input_ids], "labels": [labels]}

        target_unsloth.validate_prepared_messages_dataset(
            prepared_dataset,
            data_collator=padding_free_collator,
            max_seq_length=8,
            packing=False,
            padding_free=True,
            expected_fingerprint=expected,
        )

    def test_rejects_non_bfd_packing_and_invalid_boundaries(self):
        expected = self.fingerprint([[11, 12]])
        with self.assertRaisesRegex(RuntimeError, "requires the bfd"):
            target_unsloth.validate_prepared_messages_dataset(
                [{"input_ids": [11, 12], "seq_lengths": [2]}],
                data_collator=mock.Mock(),
                max_seq_length=8,
                packing=True,
                padding_free=True,
                packing_strategy="wrapped",
                expected_fingerprint=expected,
            )

        cases = (
            ({"input_ids": [11, 12]}, "did not produce seq_lengths"),
            (
                {"input_ids": [11, 12], "seq_lengths": [1]},
                "seq_lengths do not match",
            ),
            (
                {"input_ids": [11, 12], "seq_lengths": [0, 2]},
                "invalid sequence length",
            ),
            (
                {"input_ids": [11, 12], "seq_lengths": [True, 1]},
                "invalid sequence length",
            ),
        )
        for record, error_pattern in cases:
            with self.subTest(error_pattern=error_pattern):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.validate_prepared_messages_dataset(
                        [record],
                        data_collator=mock.Mock(),
                        max_seq_length=8,
                        packing=True,
                        padding_free=True,
                        expected_fingerprint=expected,
                    )

    def test_rejects_masks_and_non_full_sequence_labels(self):
        segment = [11, 12, 13]
        expected = self.fingerprint([segment])
        for mask_field in ("assistant_masks", "completion_mask"):
            with self.subTest(mask_field=mask_field):
                with self.assertRaisesRegex(RuntimeError, mask_field):
                    target_unsloth.validate_prepared_messages_dataset(
                        [{"input_ids": segment, mask_field: [0, 1, 1]}],
                        data_collator=mock.Mock(),
                        max_seq_length=8,
                        packing=False,
                        padding_free=False,
                        expected_fingerprint=expected,
                    )

        def masked_interior_collator(records):
            input_ids = list(records[0]["input_ids"])
            labels = list(input_ids)
            labels[1] = -100
            return {"input_ids": [input_ids], "labels": [labels]}

        with self.assertRaisesRegex(RuntimeError, "full-sequence labels"):
            target_unsloth.validate_prepared_messages_dataset(
                [{"input_ids": segment}],
                data_collator=masked_interior_collator,
                max_seq_length=8,
                packing=False,
                padding_free=False,
                expected_fingerprint=expected,
            )

    def test_rejects_one_token_segment_with_no_supervised_label(self):
        segment = [11]
        expected = self.fingerprint([segment])

        def masked_collator(records):
            return {"input_ids": [[11]], "labels": [[-100]]}

        with self.assertRaisesRegex(RuntimeError, "final tokens must be supervised"):
            target_unsloth.validate_prepared_messages_dataset(
                [{"input_ids": segment, "seq_lengths": [1]}],
                data_collator=masked_collator,
                max_seq_length=8,
                packing=True,
                padding_free=True,
                expected_fingerprint=expected,
            )

    def test_fingerprint_rejects_mutation_missing_or_extra_duplicate(self):
        first = [11, 12]
        duplicate = [21, 22]
        expected = self.fingerprint([first, duplicate, duplicate])

        def full_collator(records):
            input_ids = list(records[0]["input_ids"])
            return {"input_ids": [input_ids], "labels": [list(input_ids)]}

        cases = (
            [first, duplicate],
            [first, duplicate, duplicate, duplicate],
            [first, duplicate, [21, 99]],
        )
        for prepared_segments in cases:
            with self.subTest(prepared_segments=prepared_segments):
                prepared_dataset = [
                    {"input_ids": segment} for segment in prepared_segments
                ]
                with self.assertRaisesRegex(
                    RuntimeError,
                    "do not match the canonical conversations",
                ):
                    target_unsloth.validate_prepared_messages_dataset(
                        prepared_dataset,
                        data_collator=full_collator,
                        max_seq_length=8,
                        packing=False,
                        padding_free=False,
                        expected_fingerprint=expected,
                    )


class ResponseOnlyMessagesContractTest(unittest.TestCase):
    def fingerprint(self, segments):
        fingerprint = target_unsloth.empty_messages_token_fingerprint()
        for index, segment in enumerate(segments):
            fingerprint = target_unsloth.extend_messages_token_fingerprint(
                fingerprint,
                segment,
                description=f"test response segment {index}",
            )
        return fingerprint

    def markers(self):
        return target_unsloth.ResponseMarkers(
            instruction_part="<user>",
            response_part="<assistant>",
            instruction_token_ids=(10,),
            response_token_ids=(20,),
            use_tokenizer_parts=False,
        )

    def collator(self, records):
        record = records[0]
        return {
            "input_ids": [list(record["input_ids"])],
            "labels": [list(record["labels"])],
        }

    def test_derives_and_validates_exact_locked_marker_fixture(self):
        tokenizer = MessagesTokenizer()
        get_chat_template_parts = mock.Mock(
            return_value=("<user>", "<assistant>")
        )

        markers = target_unsloth.derive_response_markers(
            tokenizer,
            get_chat_template_parts=get_chat_template_parts,
        )

        self.assertEqual(markers.instruction_part, "<user>")
        self.assertEqual(markers.response_part, "<assistant>")
        self.assertEqual(
            markers.instruction_token_ids,
            tuple(tokenizer.raw_token_ids("<user>")),
        )
        self.assertEqual(
            markers.response_token_ids,
            tuple(tokenizer.raw_token_ids("<assistant>")),
        )
        self.assertFalse(markers.use_tokenizer_parts)
        get_chat_template_parts.assert_called_once_with(tokenizer)

    def test_validates_cached_unsloth_markers_without_passing_custom_parts(self):
        tokenizer = MessagesTokenizer()
        tokenizer._unsloth_input_part = "<user>"
        tokenizer._unsloth_output_part = "<assistant>"
        get_chat_template_parts = mock.Mock()

        markers = target_unsloth.derive_response_markers(
            tokenizer,
            get_chat_template_parts=get_chat_template_parts,
        )

        self.assertTrue(markers.use_tokenizer_parts)
        self.assertEqual(markers.instruction_part, "<user>")
        self.assertEqual(markers.response_part, "<assistant>")
        get_chat_template_parts.assert_not_called()

    def test_rejects_unusable_or_ambiguous_derived_markers(self):
        tokenizer = MessagesTokenizer()
        cases = (
            (("", "<assistant>"), "instruction marker.*non-empty"),
            (("<user>", ""), "response marker.*non-empty"),
            (("<same>", "<same>"), "markers must differ"),
            (("only-one",), "exactly two markers"),
        )
        for parts, error_pattern in cases:
            with self.subTest(parts=parts):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.derive_response_markers(
                        tokenizer,
                        get_chat_template_parts=mock.Mock(return_value=parts),
                    )

        class CollidingMarkerTokenizer(MessagesTokenizer):
            def __call__(
                self,
                text,
                *,
                add_special_tokens=True,
                truncation=False,
                max_length=None,
            ):
                return {"input_ids": [7]}

        with self.assertRaisesRegex(RuntimeError, "must tokenize differently"):
            target_unsloth.derive_response_markers(
                CollidingMarkerTokenizer(),
                get_chat_template_parts=mock.Mock(
                    return_value=("marker-a", "marker-b")
                ),
            )

        with self.assertRaisesRegex(RuntimeError, "must not contain"):
            target_unsloth.derive_response_markers(
                tokenizer,
                get_chat_template_parts=mock.Mock(
                    return_value=("<user>", "<user>x")
                ),
            )

        with self.assertRaisesRegex(RuntimeError, "could not derive stable"):
            target_unsloth.derive_response_markers(
                tokenizer,
                get_chat_template_parts=mock.Mock(
                    side_effect=ValueError("probe failed")
                ),
            )

    def test_validates_unique_marker_layout_and_rejects_collisions(self):
        markers = self.markers()
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        target_unsloth.validate_response_marker_layout(
            messages,
            [1, 10, 11, 20, 21, 2],
            markers=markers,
            subject="messages dataset row 0",
        )

        literal_collision = [
            {"role": "user", "content": "Quote <assistant> exactly."},
            {"role": "assistant", "content": "Answer."},
        ]
        with self.assertRaisesRegex(ValueError, "content collides"):
            target_unsloth.validate_response_marker_layout(
                literal_collision,
                [1, 10, 11, 20, 21, 2],
                markers=markers,
                subject="messages dataset row 0",
            )

        with self.assertRaisesRegex(ValueError, "do not uniquely match"):
            target_unsloth.validate_response_marker_layout(
                messages,
                [1, 10, 11, 20, 99, 20, 21, 2],
                markers=markers,
                subject="messages dataset row 0",
            )

    def test_response_only_rejects_role_sequences_that_unmask_non_responses(self):
        system_after_response = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
            {"role": "system", "content": "New policy."},
            {"role": "user", "content": "Follow-up?"},
            {"role": "assistant", "content": "Follow-up answer."},
        ]
        target_unsloth.validate_messages_value(
            system_after_response,
            subject="messages dataset row 0",
        )
        with self.assertRaisesRegex(ValueError, "system messages.*precede"):
            target_unsloth.validate_response_message_sequence(
                system_after_response,
                subject="messages dataset row 0",
            )

        for messages, error_pattern in (
            (
                [
                    {"role": "user", "content": "First."},
                    {"role": "user", "content": "Second."},
                    {"role": "assistant", "content": "Answer."},
                ],
                "messages to alternate",
            ),
            (
                [{"role": "assistant", "content": "Answer."}],
                "message 0 must have role 'user'",
            ),
        ):
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.validate_response_message_sequence(
                        messages,
                        subject="messages dataset row 0",
                    )

    def test_proves_prompt_masking_and_response_supervision_without_packing(self):
        markers = self.markers()
        first = [1, 10, 11, 20, 21, 22, 10, 12, 20, 23, 2]
        second = [1, 10, 13, 20, 24, 2]
        first_labels = target_unsloth.expected_response_only_labels(
            first,
            markers=markers,
        )
        second_labels = target_unsloth.expected_response_only_labels(
            second,
            markers=markers,
        )
        self.assertEqual(
            first_labels,
            [-100, -100, -100, -100, 21, 22, -100, -100, -100, 23, 2],
        )
        self.assertEqual(
            second_labels,
            [-100, -100, -100, -100, 24, 2],
        )
        prepared_dataset = [
            {"input_ids": first, "labels": first_labels},
            {"input_ids": second, "labels": second_labels},
        ]

        target_unsloth.validate_prepared_messages_dataset(
            prepared_dataset,
            data_collator=self.collator,
            max_seq_length=32,
            packing=False,
            padding_free=False,
            expected_fingerprint=self.fingerprint([first, second]),
            loss=target_unsloth.LOSS_RESPONSE,
            response_markers=markers,
        )

    def test_rejects_response_only_packing_before_inspecting_records(self):
        with self.assertRaisesRegex(RuntimeError, "does not support packing"):
            target_unsloth.validate_prepared_messages_dataset(
                [],
                data_collator=self.collator,
                max_seq_length=32,
                packing=True,
                padding_free=True,
                packing_strategy="bfd",
                expected_fingerprint=self.fingerprint([]),
                loss=target_unsloth.LOSS_RESPONSE,
                response_markers=self.markers(),
            )

    def test_rejects_full_loss_fallback_fully_masked_and_collator_changes(self):
        markers = self.markers()
        segment = [1, 10, 11, 20, 21, 2]
        expected_fingerprint = self.fingerprint([segment])
        expected_labels = target_unsloth.expected_response_only_labels(
            segment,
            markers=markers,
        )
        cases = (
            (
                "full loss fallback",
                list(segment),
                self.collator,
                "mask all non-response tokens",
            ),
            (
                "fully masked",
                [-100] * len(segment),
                self.collator,
                "response tokens must be supervised",
            ),
            (
                "collator unmasks prompt",
                expected_labels,
                lambda records: {
                    "input_ids": [list(records[0]["input_ids"])],
                    "labels": [list(records[0]["input_ids"])],
                },
                "data collator unmasked non-response tokens",
            ),
        )
        for name, labels, collator, error_pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.validate_prepared_messages_dataset(
                        [{"input_ids": segment, "labels": labels}],
                        data_collator=collator,
                        max_seq_length=16,
                        packing=False,
                        padding_free=False,
                        expected_fingerprint=expected_fingerprint,
                        loss=target_unsloth.LOSS_RESPONSE,
                        response_markers=markers,
                    )

    def test_rejects_missing_markers_or_filtered_canonical_rows(self):
        markers = self.markers()
        valid = [1, 10, 11, 20, 21, 2]
        valid_record = {
            "input_ids": valid,
            "labels": target_unsloth.expected_response_only_labels(
                valid,
                markers=markers,
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "no supervised response tokens"):
            target_unsloth.validate_prepared_messages_dataset(
                [{"input_ids": [1, 10, 11, 2], "labels": [-100] * 4}],
                data_collator=self.collator,
                max_seq_length=16,
                packing=False,
                padding_free=False,
                expected_fingerprint=self.fingerprint([[1, 10, 11, 2]]),
                loss=target_unsloth.LOSS_RESPONSE,
                response_markers=markers,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "do not match the canonical conversations",
        ):
            target_unsloth.validate_prepared_messages_dataset(
                [valid_record],
                data_collator=self.collator,
                max_seq_length=16,
                packing=False,
                padding_free=False,
                expected_fingerprint=self.fingerprint([valid, valid]),
                loss=target_unsloth.LOSS_RESPONSE,
                response_markers=markers,
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
        if dataset_type == target_unsloth.DATASET_TYPE_PREFERENCE:
            train_config = example_dpo_train_config()
        else:
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
            dependencies.dpo_trainer.assert_not_called()

        dependencies.load_dataset.assert_called_once_with(
            train_config["datasets"][0]["source"],
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
        dependencies.dpo_config.assert_not_called()
        dependencies.dpo_trainer.assert_not_called()

    def test_messages_dataset_renders_to_text_and_verifies_actual_full_loss(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        train_config["config"]["unsloth"]["packing"] = True
        duplicate = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        rows = [
            {
                "id": "conversation-0",
                "source_label": "curated",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    *duplicate,
                ],
            },
            {
                "id": "conversation-1",
                "source_label": "synthetic",
                "messages": duplicate,
            },
            {
                "id": "conversation-2",
                "source_label": "synthetic",
                "messages": duplicate,
            },
        ]
        dataset = FunctionalDataset(rows)
        dependencies = example_train_dependencies(dataset)
        fast_language_model = dependencies.fast_language_model
        base_model = fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        fast_language_model.from_pretrained.return_value = (base_model, tokenizer)
        segments = [
            tokenizer.raw_token_ids(tokenizer.render(row["messages"]))
            for row in rows
        ]
        reordered_segments = [segments[2], segments[0], segments[1]]
        prepared_record = {
            "input_ids": [
                input_id
                for segment in reordered_segments
                for input_id in segment
            ],
            "seq_lengths": [len(segment) for segment in reordered_segments],
        }
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=True,
            padding_free=True,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )
        trainer.train_dataset = [prepared_record]

        def collate(records):
            input_ids = list(records[0]["input_ids"])
            labels = list(input_ids)
            offset = 0
            for length in records[0]["seq_lengths"]:
                labels[offset] = -100
                offset += length
            return {"input_ids": [input_ids], "labels": [labels]}

        trainer.data_collator.side_effect = collate

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                target_unsloth,
                "format_alpaca_examples",
                wraps=target_unsloth.format_alpaca_examples,
            ) as format_alpaca_examples,
        ):
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        dataset.select_columns.assert_called_once_with(["messages"])
        rendered_dataset = dependencies.sft_trainer.call_args.kwargs[
            "train_dataset"
        ]
        self.assertEqual(rendered_dataset.column_names, ["text"])
        self.assertEqual(
            [row["text"] for row in rendered_dataset],
            [tokenizer.render(row["messages"]) for row in rows],
        )
        for row in rendered_dataset:
            self.assertNotIn("messages", row)
            self.assertNotIn("assistant_masks", row)
        format_alpaca_examples.assert_not_called()
        self.assertEqual(
            dependencies.sft_config.call_args.kwargs["dataset_text_field"],
            "text",
        )
        self.assertIs(
            dependencies.sft_config.call_args.kwargs["completion_only_loss"],
            False,
        )
        self.assertIs(
            dependencies.sft_config.call_args.kwargs["assistant_only_loss"],
            False,
        )
        self.assertEqual(trainer.data_collator.call_count, 1)
        trainer.train.assert_called_once_with()
        tokenizer.save_pretrained.assert_called_once()

    def test_response_loss_rejects_unsafe_role_order_before_model_allocation(self):
        cases = (
            (
                "messages",
                {
                    "messages": [
                        {"role": "user", "content": "Question?"},
                        {"role": "assistant", "content": "Answer."},
                        {"role": "system", "content": "New policy."},
                        {"role": "user", "content": "Follow-up?"},
                        {"role": "assistant", "content": "Follow-up answer."},
                    ]
                },
            ),
            (
                "sharegpt",
                {
                    "conversations": [
                        {"from": "human", "value": "Question?"},
                        {"from": "gpt", "value": "Answer."},
                        {"from": "system", "value": "New policy."},
                        {"from": "user", "value": "Follow-up?"},
                        {"from": "assistant", "value": "Follow-up answer."},
                    ]
                },
            ),
        )
        for dataset_type, row in cases:
            with self.subTest(dataset_type=dataset_type):
                train_config = example_train_config()
                train_config["datasets"][0]["type"] = dataset_type
                train_config["config"]["unsloth"]["loss"] = "response"
                dependencies = example_train_dependencies(
                    FunctionalDataset([row])
                )

                with self.assertRaisesRegex(ValueError, "system messages.*precede"):
                    target_unsloth.train_model(
                        train_config,
                        dependencies=dependencies,
                    )

                dependencies.fast_language_model.from_pretrained.assert_not_called()
                dependencies.fast_language_model.get_peft_model.assert_not_called()
                dependencies.sft_trainer.assert_not_called()

    def test_response_loss_rejects_marker_collision_before_lora_allocation(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        train_config["config"]["unsloth"]["loss"] = "response"
        dataset = FunctionalDataset(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Quote <assistant> exactly.",
                        },
                        {"role": "assistant", "content": "Answer."},
                    ]
                }
            ]
        )
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )

        with self.assertRaisesRegex(ValueError, "content collides"):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.get_chat_template_parts.assert_called_once_with(tokenizer)
        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.sft_trainer.assert_not_called()
        dependencies.train_on_responses_only.assert_not_called()

    def test_response_loss_rejects_effective_trainer_packing_before_masking(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        train_config["config"]["unsloth"]["loss"] = "response"
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = FunctionalDataset([{"messages": messages}])
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=True,
            padding_free=True,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )

        with self.assertRaisesRegex(RuntimeError, "effective trainer packing"):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.train_on_responses_only.assert_not_called()
        trainer.train.assert_not_called()

    def test_response_loss_applies_locked_unsloth_helper_and_validates_labels(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        train_config["config"]["unsloth"]["loss"] = "response"
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = FunctionalDataset([{"messages": messages}])
        dependencies = example_train_dependencies(dataset)
        fast_language_model = dependencies.fast_language_model
        base_model = fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        fast_language_model.from_pretrained.return_value = (base_model, tokenizer)
        input_ids = tokenizer.raw_token_ids(tokenizer.render(messages))
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )
        trainer.train_dataset = FunctionalDataset([{"input_ids": input_ids}])
        events = []

        def construct_trainer(**_):
            events.append("trainer constructed")
            return trainer

        def apply_response_loss(actual_trainer, **kwargs):
            events.append("response helper")
            self.assertIs(actual_trainer, trainer)
            markers = target_unsloth.ResponseMarkers(
                instruction_part=kwargs["instruction_part"],
                response_part=kwargs["response_part"],
                instruction_token_ids=tuple(
                    tokenizer.raw_token_ids(kwargs["instruction_part"])
                ),
                response_token_ids=tuple(
                    tokenizer.raw_token_ids(kwargs["response_part"])
                ),
                use_tokenizer_parts=False,
            )
            labels = target_unsloth.expected_response_only_labels(
                input_ids,
                markers=markers,
            )
            actual_trainer.train_dataset = FunctionalDataset(
                [{"input_ids": input_ids, "labels": labels}]
            )
            return actual_trainer

        def collate(records):
            return {
                "input_ids": [list(records[0]["input_ids"])],
                "labels": [list(records[0]["labels"])],
            }

        dependencies.sft_trainer.side_effect = construct_trainer
        dependencies.train_on_responses_only.side_effect = apply_response_loss
        trainer.data_collator.side_effect = collate
        trainer.train.side_effect = lambda: events.append("train")

        with tempfile.TemporaryDirectory() as temporary_directory:
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        self.assertEqual(events, ["trainer constructed", "response helper", "train"])
        dependencies.get_chat_template_parts.assert_called_once_with(tokenizer)
        dependencies.train_on_responses_only.assert_called_once_with(
            trainer,
            force_match=True,
            instruction_part="<user>",
            response_part="<assistant>",
        )
        prepared_record = trainer.train_dataset[0]
        labels = list(prepared_record["labels"])
        user_marker_start = target_unsloth.token_subsequence_index(
            input_ids,
            tokenizer.raw_token_ids("<user>"),
            start=0,
        )
        response_marker_start = target_unsloth.token_subsequence_index(
            input_ids,
            tokenizer.raw_token_ids("<assistant>"),
            start=0,
        )
        self.assertIsNotNone(user_marker_start)
        self.assertIsNotNone(response_marker_start)
        self.assertTrue(
            all(label == -100 for label in labels[:response_marker_start])
        )
        self.assertTrue(
            any(label != -100 for label in labels[response_marker_start:])
        )
        trainer.train.assert_called_once_with()

    def test_response_loss_never_falls_back_when_labels_are_fully_masked(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        train_config["config"]["unsloth"]["loss"] = "response"
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = FunctionalDataset([{"messages": messages}])
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )
        input_ids = tokenizer.raw_token_ids(tokenizer.render(messages))
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )
        trainer.train_dataset = FunctionalDataset(
            [
                {
                    "input_ids": input_ids,
                    "labels": [-100] * len(input_ids),
                }
            ]
        )
        trainer.data_collator.side_effect = lambda records: {
            "input_ids": [list(records[0]["input_ids"])],
            "labels": [list(records[0]["labels"])],
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "assistant response tokens must be supervised",
        ):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.train_on_responses_only.assert_called_once()
        trainer.train.assert_not_called()
        dependencies.fast_language_model.get_peft_model.return_value.save_pretrained.assert_not_called()
        tokenizer.save_pretrained.assert_not_called()

    def test_sharegpt_uses_canonical_messages_full_loss_pipeline(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "sharegpt"
        conversations = [
            {"from": "system", "value": "Be concise."},
            {"from": "human", "value": "Question?"},
            {"from": "gpt", "value": "Answer."},
        ]
        canonical_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = FunctionalDataset(
            [
                {
                    "id": "one-row-regression",
                    "conversations": conversations,
                }
            ]
        )
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )
        input_ids = tokenizer.raw_token_ids(tokenizer.render(canonical_messages))
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )
        trainer.train_dataset = [{"input_ids": input_ids}]
        trainer.data_collator.side_effect = lambda records: {
            "input_ids": [list(records[0]["input_ids"])],
            "labels": [list(records[0]["input_ids"])],
        }

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                target_unsloth,
                "format_alpaca_examples",
                wraps=target_unsloth.format_alpaca_examples,
            ) as format_alpaca_examples,
        ):
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        dataset.select_columns.assert_called_once_with(["conversations"])
        dataset.projected_dataset.map.assert_called_once_with(
            mock.ANY,
            batched=False,
            with_indices=True,
            remove_columns=["conversations"],
            writer_batch_size=1,
        )
        rendered_dataset = dependencies.sft_trainer.call_args.kwargs[
            "train_dataset"
        ]
        self.assertEqual(
            list(rendered_dataset),
            [{"text": tokenizer.render(canonical_messages)}],
        )
        format_alpaca_examples.assert_not_called()
        dependencies.get_chat_template_parts.assert_not_called()
        dependencies.train_on_responses_only.assert_not_called()
        for call in tokenizer.apply_calls:
            self.assertFalse(call["add_generation_prompt"])
        trainer.train.assert_called_once_with()

    def test_sharegpt_response_loss_reuses_canonical_masking_pipeline(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "sharegpt"
        train_config["config"]["unsloth"]["loss"] = "response"
        conversations = [
            {"from": "human", "value": "Question?"},
            {"from": "gpt", "value": "Answer."},
        ]
        canonical_messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        dataset = FunctionalDataset([{"conversations": conversations}])
        dependencies = example_train_dependencies(dataset)
        base_model = dependencies.fast_language_model.from_pretrained.return_value[0]
        tokenizer = MessagesTokenizer()
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )
        input_ids = tokenizer.raw_token_ids(tokenizer.render(canonical_messages))
        markers = target_unsloth.ResponseMarkers(
            instruction_part="<user>",
            response_part="<assistant>",
            instruction_token_ids=tuple(tokenizer.raw_token_ids("<user>")),
            response_token_ids=tuple(tokenizer.raw_token_ids("<assistant>")),
            use_tokenizer_parts=False,
        )
        labels = target_unsloth.expected_response_only_labels(
            input_ids,
            markers=markers,
        )
        trainer = dependencies.sft_trainer.return_value
        trainer.args = SimpleNamespace(
            packing=False,
            padding_free=False,
            packing_strategy="bfd",
            max_length=512,
            dataset_num_proc=2,
        )
        trainer.train_dataset = FunctionalDataset(
            [{"input_ids": input_ids, "labels": labels}]
        )
        trainer.data_collator.side_effect = lambda records: {
            "input_ids": [list(records[0]["input_ids"])],
            "labels": [list(records[0]["labels"])],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            target_unsloth.train_model(
                train_config,
                trained_model_directory=Path(temporary_directory)
                / "trained-model",
                dependencies=dependencies,
            )

        rendered_dataset = dependencies.sft_trainer.call_args.kwargs[
            "train_dataset"
        ]
        self.assertEqual(
            list(rendered_dataset),
            [{"text": tokenizer.render(canonical_messages)}],
        )
        dependencies.get_chat_template_parts.assert_called_once_with(tokenizer)
        dependencies.train_on_responses_only.assert_called_once_with(
            trainer,
            force_match=True,
            instruction_part="<user>",
            response_part="<assistant>",
        )
        self.assertTrue(any(label != -100 for label in labels))
        trainer.train.assert_called_once_with()

    def test_wall_clock_template_check_precedes_rendering_and_lora(self):
        train_config = example_train_config()
        train_config["datasets"][0]["type"] = "messages"
        dataset = FunctionalDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Question?"},
                        {"role": "assistant", "content": "Answer."},
                    ]
                }
            ]
        )
        dependencies = example_train_dependencies(dataset)
        base_model = (
            dependencies.fast_language_model.from_pretrained.return_value[0]
        )
        tokenizer = MessagesTokenizer(
            chat_template="{{ strftime_now('%Y-%m-%d') }}{{ messages }}"
        )
        dependencies.fast_language_model.from_pretrained.return_value = (
            base_model,
            tokenizer,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"wall-clock-dependent.*strftime_now",
        ):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dataset.projected_dataset.map.assert_not_called()
        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.resolve_model_name.assert_not_called()
        dependencies.model_info.assert_not_called()
        dependencies.sft_trainer.assert_not_called()

    def test_messages_template_render_and_token_checks_precede_lora(self):
        valid_row = {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
        cases = (
            (
                "missing template",
                MessagesTokenizer(chat_template=None),
                512,
                "usable tokenizer chat template",
            ),
            (
                "render failure",
                MessagesTokenizer(),
                512,
                "could not be rendered",
            ),
            (
                "token mismatch",
                MessagesTokenizer(),
                512,
                "canonical chat-template token IDs do not match",
            ),
            (
                "overflow",
                MessagesTokenizer(),
                2,
                "exceeding maxSeqLength 2",
            ),
        )
        cases[1][1].render_error = RuntimeError("template failure")
        cases[2][1].mutate_rendered_tokenization = True

        for name, tokenizer, max_seq_length, error_pattern in cases:
            with self.subTest(name=name):
                train_config = example_train_config()
                train_config["datasets"][0]["type"] = "messages"
                train_config["config"]["unsloth"][
                    "maxSeqLength"
                ] = max_seq_length
                dataset = FunctionalDataset([valid_row])
                dependencies = example_train_dependencies(dataset)
                base_model = (
                    dependencies.fast_language_model.from_pretrained.return_value[0]
                )
                dependencies.fast_language_model.from_pretrained.return_value = (
                    base_model,
                    tokenizer,
                )

                with self.assertRaisesRegex(
                    (RuntimeError, ValueError),
                    error_pattern,
                ):
                    target_unsloth.train_model(
                        train_config,
                        dependencies=dependencies,
                    )

                dependencies.fast_language_model.get_peft_model.assert_not_called()
                dependencies.resolve_model_name.assert_not_called()
                dependencies.model_info.assert_not_called()
                dependencies.sft_trainer.assert_not_called()

    def test_messages_loader_failure_redacts_signed_url_and_cause(self):
        signed_url = (
            "https://user:password@example.com/messages.jsonl"
            "?X-Amz-Credential=secret&token=private#fragment"
        )
        train_config = example_train_config()
        train_config["datasets"][0].update(
            {"type": "messages", "source": signed_url}
        )
        dependencies = example_train_dependencies(mock.Mock())
        dependencies.load_dataset.side_effect = RuntimeError(
            f"failed to load {signed_url}"
        )

        local_file = Path("/cache/content.jsonl")
        with mock.patch.object(
            target_unsloth,
            "materialize_remote_dataset_file",
            return_value=nullcontext(local_file),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "messages dataset remote JSON URL could not be loaded",
            ) as raised:
                target_unsloth.train_model(
                    train_config,
                    dependencies=dependencies,
                )

        message = str(raised.exception)
        self.assertNotIn("user", message)
        self.assertNotIn("password", message)
        self.assertNotIn("secret", message)
        self.assertNotIn("private", message)
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)
        dependencies.load_dataset.assert_called_once_with(
            "json",
            data_files={"train": str(local_file)},
            split="train",
        )
        dependencies.fast_language_model.from_pretrained.assert_not_called()
        dependencies.fast_language_model.get_peft_model.assert_not_called()

    def test_checksum_failure_prevents_parser_and_model_allocation(self):
        body = b'{"messages":[{"role":"user","content":"Q"}]}\n'
        with LocalDatasetServer({"/messages.jsonl": body}) as server:
            signed_url = (
                server.url("/messages.jsonl")
                + "?token=private-value#fragment-marker"
            )
            train_config = example_train_config()
            train_config["datasets"][0].update(
                {
                    "type": "messages",
                    "source": signed_url,
                    "loader": {
                        "type": "json",
                        "checksum": "sha256:" + "0" * 64,
                    },
                }
            )
            dependencies = example_train_dependencies(mock.Mock())

            with tempfile.TemporaryDirectory() as cache_directory:
                with mock.patch.dict(
                    target_unsloth.os.environ,
                    {"HF_DATASETS_CACHE": cache_directory},
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "checksum does not match",
                    ) as raised:
                        target_unsloth.train_model(
                            train_config,
                            dependencies=dependencies,
                        )

        for secret in (
            "token",
            "private-value",
            "fragment-marker",
        ):
            self.assertNotIn(secret, str(raised.exception))
        dependencies.load_dataset.assert_not_called()
        dependencies.fast_language_model.from_pretrained.assert_not_called()
        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.resolve_model_name.assert_not_called()
        dependencies.model_info.assert_not_called()
        dependencies.sft_trainer.assert_not_called()

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

    def test_rejects_invalid_loss_before_loading_or_model_allocation(self):
        cases = (
            (
                "alpaca",
                "response",
                False,
                "supported only for messages and sharegpt",
            ),
            ("messages", "assistant", False, "unsupported SFT loss"),
            ("messages", "response", True, "does not support packing"),
        )
        for dataset_type, loss, packing, error_pattern in cases:
            with self.subTest(
                dataset_type=dataset_type,
                loss=loss,
                packing=packing,
            ):
                train_config = example_train_config()
                train_config["datasets"][0]["type"] = dataset_type
                train_config["config"]["unsloth"]["loss"] = loss
                train_config["config"]["unsloth"]["packing"] = packing
                dependencies = example_train_dependencies(mock.Mock())

                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.train_model(
                        train_config,
                        dependencies=dependencies,
                    )

                dependencies.load_dataset.assert_not_called()
                dependencies.fast_language_model.from_pretrained.assert_not_called()
                dependencies.sft_trainer.assert_not_called()
                dependencies.train_on_responses_only.assert_not_called()

    def test_rejects_empty_or_invalid_datasets_before_model_allocation(self):
        valid_prompt_completion = {
            "prompt": "Question?",
            "completion": " Answer.",
        }
        valid_messages = {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
        valid_text = {"text": "A complete preformatted sequence."}
        valid_preference = {
            "prompt": "Question?",
            "chosen": "A careful answer.",
            "rejected": "An unsafe answer.",
        }
        cases = (
            (
                "empty messages dataset",
                "messages",
                [],
                "messages dataset source 'organization/dataset'.*at least one",
            ),
            (
                "missing messages column",
                "messages",
                [{"content": "Not a conversation."}],
                "messages dataset source 'organization/dataset'.*"
                "missing required columns.*messages",
            ),
            (
                "invalid messages value",
                "messages",
                [{"messages": "not a list"}],
                "messages.*non-empty list",
            ),
            (
                "messages metadata",
                "messages",
                [{**valid_messages, "tools": []}],
                "unsupported top-level fields.*tools",
            ),
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
                "empty preference dataset",
                "preference",
                [],
                "preference dataset source 'organization/preferences'.*at least one",
            ),
            (
                "missing preference column",
                "preference",
                [{"prompt": "Question?", "chosen": "Answer."}],
                "missing required columns.*rejected",
            ),
            (
                "null preference prompt",
                "preference",
                [{**valid_preference, "prompt": None}],
                "prompt.*string",
            ),
            (
                "non-string preference chosen",
                "preference",
                [{**valid_preference, "chosen": ["Answer"]}],
                "chosen.*string",
            ),
            (
                "null preference rejected",
                "preference",
                [{**valid_preference, "rejected": None}],
                "rejected.*string",
            ),
            (
                "empty preference prompt",
                "preference",
                [{**valid_preference, "prompt": ""}],
                "prompt.*non-empty string",
            ),
            (
                "whitespace preference chosen",
                "preference",
                [{**valid_preference, "chosen": " \t"}],
                "chosen.*non-empty string",
            ),
            (
                "empty preference rejected",
                "preference",
                [{**valid_preference, "rejected": ""}],
                "rejected.*non-empty string",
            ),
            (
                "identical preference choices",
                "preference",
                [{**valid_preference, "rejected": valid_preference["chosen"]}],
                "chosen.*rejected.*distinct",
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
            dpo_config=mock.Mock(),
            dpo_trainer=mock.Mock(),
            get_chat_template_parts=mock.Mock(),
            train_on_responses_only=mock.Mock(),
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


class DPOTrainingPhaseTest(unittest.TestCase):
    def preference_rows(self):
        return [
            {
                "prompt": "How should I rotate an API key?",
                "chosen": "Deploy a replacement before revoking the old key.",
                "rejected": "Revoke the old key before creating a replacement.",
                "metadata": "must be removed",
            },
            {
                "prompt": "How should I store a secret?",
                "chosen": "Use a dedicated secret manager.",
                "rejected": "Commit it to source control.",
                "metadata": "must also be removed",
            },
        ]

    def test_trains_dpo_without_sft_formatting_and_preserves_preferences(self):
        train_config = example_dpo_train_config()
        train_config["objective"].update(beta=0.25, maxPromptLength=128)
        train_config["config"]["unsloth"].update(
            maxSeqLength=1024,
            batchSize=1,
            gradientAccumulationSteps=2,
            warmupSteps=3,
            maxSteps=4,
            learningRate=0.000001,
            loggingSteps=5,
            optimizer="adamw_8bit",
            weightDecay=0.02,
            lrSchedulerType="cosine",
            seed=7,
        )
        dataset = FunctionalDataset(self.preference_rows())
        dependencies = example_dpo_train_dependencies(
            dataset,
            train_config=train_config,
        )
        base_model, tokenizer = (
            dependencies.fast_language_model.from_pretrained.return_value
        )
        adapter_model = (
            dependencies.fast_language_model.get_peft_model.return_value
        )
        runtime_trainer = dependencies.dpo_trainer.runtime_trainer

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                target_unsloth,
                "prepare_training_dataset",
            ) as prepare_training_dataset,
            mock.patch.object(
                target_unsloth,
                "format_alpaca_examples",
            ) as format_alpaca_examples,
        ):
            trained_model_directory = (
                Path(temporary_directory) / "trained-model"
            )
            result = target_unsloth.train_model(
                train_config,
                trained_model_directory=trained_model_directory,
                dependencies=dependencies,
            )

        self.assertEqual(result, trained_model_directory)
        dependencies.load_dataset.assert_called_once_with(
            "organization/preferences",
            split="train",
        )
        dataset.select_columns.assert_called_once_with(
            ["prompt", "chosen", "rejected"]
        )
        projected_dataset = dataset.projected_dataset
        self.assertEqual(
            projected_dataset.column_names,
            ["prompt", "chosen", "rejected"],
        )
        self.assertEqual(
            list(projected_dataset),
            [
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
                for row in self.preference_rows()
            ],
        )
        projected_dataset.map.assert_not_called()
        prepare_training_dataset.assert_not_called()
        format_alpaca_examples.assert_not_called()
        dependencies.sft_config.assert_not_called()
        dependencies.sft_trainer.assert_not_called()
        dependencies.get_chat_template_parts.assert_not_called()
        dependencies.train_on_responses_only.assert_not_called()

        dependencies.dpo_config.assert_called_once_with(
            output_dir="outputs",
            dataset_num_proc=2,
            max_length=1024,
            max_prompt_length=128,
            max_completion_length=None,
            truncation_mode="keep_end",
            beta=0.25,
            loss_type="sigmoid",
            reference_free=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=3,
            max_steps=4,
            learning_rate=0.000001,
            fp16=False,
            bf16=True,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.02,
            lr_scheduler_type="cosine",
            seed=7,
            save_strategy="no",
            report_to="none",
        )
        dependencies.dpo_trainer.assert_called_once_with(
            model=adapter_model,
            ref_model=None,
            train_dataset=projected_dataset,
            processing_class=tokenizer,
            args="dpo-config",
        )
        dpo_kwargs = dependencies.dpo_config.call_args.kwargs
        for sft_only_field in (
            "packing",
            "dataset_text_field",
            "assistant_only_loss",
            "completion_only_loss",
            "peft_config",
        ):
            self.assertNotIn(sft_only_field, dpo_kwargs)
            self.assertNotIn(
                sft_only_field,
                dependencies.dpo_trainer.call_args.kwargs,
            )

        runtime_trainer.train.assert_called_once_with()
        dependencies.fast_language_model.from_pretrained.assert_called_once_with(
            model_name="example/model",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        dependencies.fast_language_model.get_peft_model.assert_called_once()
        adapter_model.save_pretrained.assert_called_once_with(
            trained_model_directory
        )
        tokenizer.save_pretrained.assert_called_once_with(
            trained_model_directory
        )
        self.assertIs(
            base_model,
            dependencies.fast_language_model.from_pretrained.return_value[0],
        )

    def test_rejects_dpo_configuration_before_loading_or_model_allocation(self):
        cases = (
            (
                "SFT dataset",
                lambda config: config["datasets"][0].update(type="alpaca"),
                "DPO objective requires a preference dataset",
            ),
            (
                "packing",
                lambda config: config["config"]["unsloth"].update(
                    packing=True
                ),
                "does not support packing",
            ),
            (
                "response loss",
                lambda config: config["config"]["unsloth"].update(
                    loss="response"
                ),
                "response SFT loss",
            ),
        )
        for name, mutate, error_pattern in cases:
            with self.subTest(name=name):
                train_config = example_dpo_train_config()
                mutate(train_config)
                dependencies = example_train_dependencies(mock.Mock())
                with self.assertRaisesRegex(ValueError, error_pattern):
                    target_unsloth.train_model(
                        train_config,
                        dependencies=dependencies,
                    )
                dependencies.load_dataset.assert_not_called()
                dependencies.fast_language_model.from_pretrained.assert_not_called()
                dependencies.fast_language_model.get_peft_model.assert_not_called()
                dependencies.sft_trainer.assert_not_called()
                dependencies.dpo_trainer.assert_not_called()

    def test_rejects_invalid_dpo_runtime_contract_before_training(self):
        cases = (
            (
                "reference model",
                lambda trainer, _model: setattr(trainer, "ref_model", object()),
                "must use ref_model=None",
            ),
            (
                "not PEFT",
                lambda trainer, _model: setattr(
                    trainer, "is_peft_model", False
                ),
                "did not recognize.*PEFT",
            ),
            (
                "reference free",
                lambda trainer, _model: setattr(
                    trainer, "reference_free", True
                ),
                "must not use reference-free",
            ),
            (
                "policy model replaced",
                lambda trainer, _model: setattr(trainer, "model", mock.Mock()),
                "replaced the policy model",
            ),
            (
                "adapter cannot be disabled",
                lambda _trainer, model: setattr(model, "disable_adapter", None),
                "does not support disabling.*PEFT adapter",
            ),
            (
                "beta mismatch",
                lambda trainer, _model: setattr(trainer, "beta", 0.2),
                "beta does not match",
            ),
            (
                "loss mismatch",
                lambda trainer, _model: setattr(
                    trainer, "loss_type", ["hinge"]
                ),
                "loss type does not match",
            ),
            (
                "prompt length mismatch",
                lambda trainer, _model: setattr(
                    trainer, "max_prompt_length", 128
                ),
                "max prompt length does not match",
            ),
            (
                "sequence length mismatch",
                lambda trainer, _model: setattr(
                    trainer, "max_length", 1024
                ),
                "max length does not match",
            ),
        )
        for name, mutate, error_pattern in cases:
            with self.subTest(name=name):
                train_config = example_dpo_train_config()
                dataset = in_memory_dataset(self.preference_rows())
                dependencies = example_dpo_train_dependencies(dataset)
                runtime_trainer = mock.Mock()
                runtime_trainer.ref_model = None
                runtime_trainer.is_peft_model = True
                runtime_trainer.reference_free = False
                runtime_trainer.beta = 0.1
                runtime_trainer.loss_type = ["sigmoid"]
                runtime_trainer.max_prompt_length = 512
                runtime_trainer.max_length = 2048
                runtime_trainer.model = (
                    dependencies.fast_language_model.get_peft_model.return_value
                )
                runtime_trainer.accelerator = mock.Mock()
                runtime_trainer.accelerator.unwrap_model.side_effect = (
                    lambda model: model
                )

                def construct_trainer(**kwargs):
                    runtime_trainer.train_dataset = kwargs["train_dataset"]
                    return runtime_trainer

                dependencies.dpo_trainer.side_effect = construct_trainer
                mutate(runtime_trainer, runtime_trainer.model)
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    target_unsloth.train_model(
                        train_config,
                        dependencies=dependencies,
                    )
                runtime_trainer.train.assert_not_called()
                runtime_trainer.model.save_pretrained.assert_not_called()

    def test_rejects_empty_dataset_after_dpo_trainer_preprocessing(self):
        train_config = example_dpo_train_config()
        dataset = in_memory_dataset(self.preference_rows())
        dependencies = example_dpo_train_dependencies(dataset)
        runtime_trainer = dependencies.dpo_trainer.runtime_trainer

        def construct_empty_trainer(**_kwargs):
            runtime_trainer.train_dataset = []
            return runtime_trainer

        dependencies.dpo_trainer.side_effect = construct_empty_trainer
        with self.assertRaisesRegex(RuntimeError, "prepared an empty"):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        runtime_trainer.train.assert_not_called()
        runtime_trainer.model.save_pretrained.assert_not_called()

    def test_dpo_checksum_failure_prevents_parser_and_model_allocation(self):
        body = (
            b'{"prompt":"Question?","chosen":"Safe",'
            b'"rejected":"Unsafe"}\n'
        )
        with LocalDatasetServer({"/preferences.jsonl": body}) as server:
            signed_url = (
                server.url("/preferences.jsonl")
                + "?token=private-value#fragment-marker"
            )
            train_config = example_dpo_train_config()
            train_config["datasets"][0].update(
                source=signed_url,
                loader={
                    "type": "json",
                    "checksum": "sha256:" + "0" * 64,
                },
            )
            dependencies = example_dpo_train_dependencies(mock.Mock())

            with tempfile.TemporaryDirectory() as cache_directory:
                with mock.patch.dict(
                    target_unsloth.os.environ,
                    {"HF_DATASETS_CACHE": cache_directory},
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "checksum does not match",
                    ) as raised:
                        target_unsloth.train_model(
                            train_config,
                            dependencies=dependencies,
                        )

        for secret in ("token", "private-value", "fragment-marker"):
            self.assertNotIn(secret, str(raised.exception))
        dependencies.load_dataset.assert_not_called()
        dependencies.fast_language_model.from_pretrained.assert_not_called()
        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.dpo_trainer.assert_not_called()

    def test_rejects_missing_dpo_tokenizer_eos_before_lora_allocation(self):
        train_config = example_dpo_train_config()
        dataset = in_memory_dataset(self.preference_rows())
        dependencies = example_dpo_train_dependencies(dataset)
        tokenizer = dependencies.fast_language_model.from_pretrained.return_value[1]
        tokenizer.eos_token = None

        with self.assertRaisesRegex(RuntimeError, "tokenizer EOS token"):
            target_unsloth.train_model(
                train_config,
                dependencies=dependencies,
            )

        dependencies.fast_language_model.get_peft_model.assert_not_called()
        dependencies.dpo_trainer.assert_not_called()

    def test_imports_unsloth_before_trl_dpo_dependencies(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        function_start = source.index("def load_train_dependencies()")
        function_end = source.index("\ndef load_export_dependencies()", function_start)
        dependency_source = source[function_start:function_end]
        self.assertLess(
            dependency_source.index("from unsloth import"),
            dependency_source.index("from trl import"),
        )
        self.assertIn("DPOConfig", dependency_source)
        self.assertIn("DPOTrainer", dependency_source)

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
