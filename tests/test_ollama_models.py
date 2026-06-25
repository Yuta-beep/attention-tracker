import unittest

from pathlib import Path
import tempfile

from last_token_attention.ollama_cli import (
    _load_cases,
    _summarize_results,
    _tokens_per_second,
)
from last_token_attention.ollama_models import resolve_ollama_models


class OllamaModelRegistryTest(unittest.TestCase):
    def test_resolve_all_models(self) -> None:
        models = resolve_ollama_models("all")
        self.assertEqual(
            [model.model for model in models],
            ["qwen3.5:9b", "gemma4:12b", "gpt-oss:20b"],
        )

    def test_resolve_registry_key_and_exact_tag(self) -> None:
        models = resolve_ollama_models("qwen3.5_9b,gpt-oss:20b")
        self.assertEqual(
            [model.model for model in models],
            ["qwen3.5:9b", "gpt-oss:20b"],
        )

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_ollama_models("unknown")

    def test_tokens_per_second(self) -> None:
        self.assertEqual(_tokens_per_second(100, 2_000_000_000), 50.0)
        self.assertIsNone(_tokens_per_second(0, 2_000_000_000))

    def test_load_cases_reads_attention_tracker_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"id":"case-1","instruction":"Classify.",'
                '"normal_text":"clean","injected_text":"attack"}\n',
                encoding="utf-8",
            )
            cases = _load_cases(str(path))
        self.assertEqual(cases[0]["normal_text"], "clean")
        self.assertEqual(cases[0]["injected_text"], "attack")

    def test_summary_groups_normal_and_attack_inferences(self) -> None:
        rows = [
            {
                "model": "model-a",
                "wall_seconds": 2.0,
                "prompt_tokens_per_second": 100.0,
                "generation_tokens_per_second": 20.0,
            },
            {
                "model": "model-a",
                "wall_seconds": 4.0,
                "prompt_tokens_per_second": 200.0,
                "generation_tokens_per_second": 40.0,
            },
        ]
        summary = _summarize_results(rows)
        self.assertEqual(summary[0]["num_inferences"], 2)
        self.assertEqual(summary[0]["mean_wall_seconds"], 3.0)
        self.assertEqual(summary[0]["mean_generation_tokens_per_second"], 30.0)


if __name__ == "__main__":
    unittest.main()
