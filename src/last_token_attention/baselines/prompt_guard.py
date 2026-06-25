import torch

from .classifier import SequenceClassifierDetector, resolve_label_index


DEFAULT_MODEL_ID = "meta-llama/Prompt-Guard-86M"


class PromptGuardDetector(SequenceClassifierDetector):
    name = "prompt_guard"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_length: int = 512):
        super().__init__(model_id=model_id, max_length=max_length)
        self.injection_index = resolve_label_index(
            self.bundle.model,
            ("injection", "prompt_injection", "label_1"),
        )
        self.jailbreak_index = resolve_label_index(
            self.bundle.model,
            ("jailbreak", "label_2"),
        )

    def attack_score_from_logits(self, logits: torch.Tensor) -> float:
        # This follows the Attention Tracker paper: injection logit + jailbreak logit.
        return float(
            (logits[self.injection_index] + logits[self.jailbreak_index]).item()
        )


def main() -> None:
    from .cli import main_for_detector

    main_for_detector("prompt_guard")
