from dataclasses import dataclass
import os

from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import ModelConfig


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    config: ModelConfig


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype={dtype_name!r}") from exc


def _resolve_hf_token() -> str | None:
    load_dotenv(".env.local", override=False)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_model_bundle(config: ModelConfig) -> ModelBundle:
    hf_token = _resolve_hf_token()
    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": config.trust_remote_code,
    }
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_torch_dtype(config.torch_dtype)
    model_kwargs = {
        "device_map": "auto",
        "output_attentions": True,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if hf_token:
        model_kwargs["token"] = hf_token

    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    loaders = {
        "causal_lm": AutoModelForCausalLM,
        "image_text_to_text": AutoModelForImageTextToText,
        "multimodal_lm": AutoModelForMultimodalLM,
    }
    try:
        loader = loaders[config.model_loader]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_loader={config.model_loader!r}") from exc
    model = loader.from_pretrained(config.model_id, **model_kwargs)
    model.eval()

    device = next(model.parameters()).device
    return ModelBundle(tokenizer=tokenizer, model=model, device=device, config=config)
