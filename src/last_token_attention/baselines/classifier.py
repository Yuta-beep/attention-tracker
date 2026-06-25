from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..modeling import _resolve_hf_token
from .common import DetectionExample


@dataclass
class ClassifierBundle:
    tokenizer: object
    model: object
    device: torch.device


def load_classifier_bundle(model_id: str) -> ClassifierBundle:
    token = _resolve_hf_token()
    kwargs = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        device_map="auto",
        **kwargs,
    )
    model.eval()
    return ClassifierBundle(
        tokenizer=tokenizer,
        model=model,
        device=next(model.parameters()).device,
    )


def resolve_label_index(model, candidates: tuple[str, ...]) -> int:
    labels = {
        int(index): str(label).lower().replace("-", "_").replace(" ", "_")
        for index, label in model.config.id2label.items()
    }
    normalized_candidates = {
        candidate.lower().replace("-", "_").replace(" ", "_")
        for candidate in candidates
    }
    for index, label in labels.items():
        if label in normalized_candidates:
            return index
    raise ValueError(
        f"Could not find any of labels {sorted(normalized_candidates)} "
        f"in model id2label={labels}."
    )


class SequenceClassifierDetector:
    name = "sequence_classifier"

    def __init__(self, model_id: str, max_length: int = 512):
        self.model_id = model_id
        self.max_length = max_length
        self.bundle = load_classifier_bundle(model_id)

    def score(self, example: DetectionExample) -> float:
        inputs = self.bundle.tokenizer(
            example.text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.bundle.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.bundle.model(**inputs).logits[0]
        return self.attack_score_from_logits(logits)

    def attack_score_from_logits(self, logits: torch.Tensor) -> float:
        raise NotImplementedError
