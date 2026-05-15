from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_id: str
    uses_chat_template: bool = True
    system_role_supported: bool = True
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"


MODEL_REGISTRY = {
    "qwen2_1.5b": ModelConfig(
        key="qwen2_1.5b",
        model_id="Qwen/Qwen2-1.5B-Instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=False,
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
    "phi3_mini": ModelConfig(
        key="phi3_mini",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        uses_chat_template=True,
        system_role_supported=True,
        load_in_4bit=False,
        torch_dtype="bfloat16",
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_key={model_key!r}. Known keys: {known}") from exc
