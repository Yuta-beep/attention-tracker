from dataclasses import dataclass
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import ModelConfig
from ..modeling import _resolve_hf_token, _resolve_torch_dtype
from .common import DetectionExample


@dataclass
class CausalBundle:
    tokenizer: object
    model: object
    device: torch.device


def load_causal_bundle(config: ModelConfig) -> CausalBundle:
    token = _resolve_hf_token()
    auth = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=True,
        **auth,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_torch_dtype(config.torch_dtype)
    model_kwargs = {"device_map": "auto", **auth}
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    model.eval()
    return CausalBundle(
        tokenizer=tokenizer,
        model=model,
        device=next(model.parameters()).device,
    )


def build_generation_prompt(tokenizer, content: str, uses_chat_template: bool) -> str:
    if not uses_chat_template:
        return content
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def completion_log_probability(
    bundle: CausalBundle,
    prompt: str,
    completion: str,
) -> float:
    prompt_ids = bundle.tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    completion_ids = bundle.tokenizer(
        completion,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    if completion_ids.shape[1] == 0:
        raise ValueError("Completion must contain at least one token.")

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(bundle.device)
    logits = bundle.model(input_ids=input_ids, use_cache=False).logits[0]
    prompt_length = prompt_ids.shape[1]
    total = 0.0
    for offset, token_id in enumerate(completion_ids[0].tolist()):
        prediction_position = prompt_length + offset - 1
        token_log_probs = torch.log_softmax(logits[prediction_position], dim=-1)
        total += float(token_log_probs[token_id].item())
    return total


class CausalProbabilityDetector:
    name = "causal_probability"

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.model_id
        self.bundle = load_causal_bundle(config)

    def prompt_content(self, example: DetectionExample) -> str:
        raise NotImplementedError

    def attack_score_from_log_probability(self, log_probability: float) -> float:
        return math.exp(log_probability)

    def score(self, example: DetectionExample) -> float:
        content = self.prompt_content(example)
        prompt = build_generation_prompt(
            self.bundle.tokenizer,
            content,
            self.config.uses_chat_template,
        )
        candidates = self.completion_candidates()
        log_probability = max(
            completion_log_probability(self.bundle, prompt, candidate)
            for candidate in candidates
        )
        return self.attack_score_from_log_probability(log_probability)

    def completion_candidates(self) -> tuple[str, ...]:
        raise NotImplementedError
