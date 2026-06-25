import unittest

from last_token_attention.ollama_cli import _tokens_per_second
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


if __name__ == "__main__":
    unittest.main()
