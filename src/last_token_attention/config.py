from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_id: str
    uses_chat_template: bool = True
    system_role_supported: bool = True
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    model_loader: str = "causal_lm"
    trust_remote_code: bool = False
    attn_implementation: str = ""
    reasoning_enabled: bool = False
    attention_layer_type: str = ""


MODEL_REGISTRY = {
    "qwen2_1.5b": ModelConfig(
        key="qwen2_1.5b",
        model_id="Qwen/Qwen2-1.5B-Instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=False,
        torch_dtype="bfloat16",
    ),
    "qwen2_7b": ModelConfig(
        key="qwen2_7b",
        model_id="Qwen/Qwen2-7B-Instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=True,
        torch_dtype="bfloat16",
    ),
    "llama3_8b": ModelConfig(
        key="llama3_8b",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=True,
        torch_dtype="bfloat16",
    ),
    "mistral_7b": ModelConfig(
        key="mistral_7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=True,
        torch_dtype="bfloat16",
    ),
    "phi3_mini": ModelConfig(
        key="phi3_mini",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=False,
        torch_dtype="bfloat16",
    ),
    "gemma2_2b": ModelConfig(
        key="gemma2_2b",
        model_id="google/gemma-2-2b-it",
        uses_chat_template=True,
        system_role_supported=False,
        load_in_4bit=False,
        torch_dtype="bfloat16",
    ),
    "gemma2_9b": ModelConfig(
        key="gemma2_9b",
        model_id="google/gemma-2-9b-it",
        uses_chat_template=True,
        system_role_supported=False,
        load_in_4bit=True,
        torch_dtype="bfloat16",
    ),
    "qwen3.5_9b": ModelConfig(
        key="qwen3.5_9b",
        model_id="Qwen/Qwen3.5-9B",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=True,
        torch_dtype="bfloat16",
        model_loader="image_text_to_text",
        trust_remote_code=True,
        attn_implementation="eager",
        reasoning_enabled=True,
        attention_layer_type="full_attention",
    ),
    "gemma4_12b": ModelConfig(
        key="gemma4_12b",
        model_id="google/gemma-4-12B-it",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=True,
        torch_dtype="bfloat16",
        model_loader="multimodal_lm",
        trust_remote_code=True,
        attn_implementation="eager",
        reasoning_enabled=True,
        attention_layer_type="full_attention",
    ),
    "gpt_oss_20b": ModelConfig(
        key="gpt_oss_20b",
        model_id="openai/gpt-oss-20b",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=False,
        torch_dtype="bfloat16",
        model_loader="causal_lm",
        trust_remote_code=True,
        attn_implementation="eager",
        reasoning_enabled=True,
        attention_layer_type="full_attention",
    ),
}


MODEL_KEY_PATTERN = re.compile(r"[^a-z0-9._-]+")


def _model_key_from_id(model_id: str) -> str:
    normalized = model_id.strip().lower().replace("/", "-").replace(" ", "-")
    normalized = MODEL_KEY_PATTERN.sub("-", normalized)
    normalized = normalized.strip("-._")
    return normalized or "custom-model"


def get_model_config(model_key: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_key={model_key!r}. Known keys: {known}") from exc


def resolve_model_config(
    model: str,
    *,
    torch_dtype: str | None = None,
    load_in_4bit: bool | None = None,
    uses_chat_template: bool | None = None,
    system_role_supported: bool | None = None,
) -> ModelConfig:
    if model in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model]
    else:
        config = ModelConfig(
            key=_model_key_from_id(model),
            model_id=model,
        )

    overrides = {}
    if torch_dtype is not None:
        overrides["torch_dtype"] = torch_dtype
    if load_in_4bit is not None:
        overrides["load_in_4bit"] = load_in_4bit
    if uses_chat_template is not None:
        overrides["uses_chat_template"] = uses_chat_template
    if system_role_supported is not None:
        overrides["system_role_supported"] = system_role_supported
    if overrides:
        config = replace(config, **overrides)
    return config
