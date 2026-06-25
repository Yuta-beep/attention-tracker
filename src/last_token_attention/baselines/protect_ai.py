import torch

from .classifier import SequenceClassifierDetector, resolve_label_index


DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"


class ProtectAIDetector(SequenceClassifierDetector):
    name = "protect_ai"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_length: int = 512):
        super().__init__(model_id=model_id, max_length=max_length)
        self.injection_index = resolve_label_index(
            self.bundle.model,
            ("injection", "prompt_injection", "injection_detected", "label_1"),
        )

    def attack_score_from_logits(self, logits: torch.Tensor) -> float:
        return float(logits[self.injection_index].item())


def main() -> None:
    from .cli import main_for_detector

    main_for_detector("protect_ai")


if __name__ == "__main__":
    main()
