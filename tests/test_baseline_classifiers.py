from types import SimpleNamespace

import torch

from last_token_attention.baselines.classifier import resolve_label_index


def test_resolve_label_index_normalizes_model_labels():
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}
        )
    )

    assert resolve_label_index(model, ("prompt_injection", "injection")) == 1
    assert resolve_label_index(model, ("jailbreak",)) == 2


def test_prompt_guard_score_adds_attack_logits_without_loading_model():
    from last_token_attention.baselines.prompt_guard import PromptGuardDetector

    detector = object.__new__(PromptGuardDetector)
    detector.injection_index = 1
    detector.jailbreak_index = 2

    assert detector.attack_score_from_logits(torch.tensor([1.0, 2.0, 3.0])) == 5.0


def test_protect_ai_score_uses_injection_logit_without_loading_model():
    from last_token_attention.baselines.protect_ai import ProtectAIDetector

    detector = object.__new__(ProtectAIDetector)
    detector.injection_index = 1

    assert detector.attack_score_from_logits(torch.tensor([1.0, 2.5])) == 2.5
