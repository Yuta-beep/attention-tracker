from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaModelConfig:
    key: str
    model: str
    role: str
    default_context: int
    expected_download_gb: float


OLLAMA_MODEL_REGISTRY = {
    "qwen3.5_9b": OllamaModelConfig(
        key="qwen3.5_9b",
        model="qwen3.5:9b",
        role="General reasoning, Japanese, coding, vision, and tool use",
        default_context=32768,
        expected_download_gb=6.6,
    ),
    "gemma4_12b": OllamaModelConfig(
        key="gemma4_12b",
        model="gemma4:12b",
        role="Independent multimodal reasoning and coding comparison",
        default_context=32768,
        expected_download_gb=7.6,
    ),
    "gpt_oss_20b": OllamaModelConfig(
        key="gpt_oss_20b",
        model="gpt-oss:20b",
        role="Reasoning, structured output, coding, and agentic tool use",
        default_context=16384,
        expected_download_gb=14.0,
    ),
}


DEFAULT_OLLAMA_MODELS = list(OLLAMA_MODEL_REGISTRY)


def resolve_ollama_models(raw_models: str) -> list[OllamaModelConfig]:
    if raw_models.strip().lower() == "all":
        return [OLLAMA_MODEL_REGISTRY[key] for key in DEFAULT_OLLAMA_MODELS]

    resolved = []
    for item in raw_models.split(","):
        value = item.strip()
        if not value:
            continue
        if value in OLLAMA_MODEL_REGISTRY:
            resolved.append(OLLAMA_MODEL_REGISTRY[value])
            continue

        matches = [
            config
            for config in OLLAMA_MODEL_REGISTRY.values()
            if config.model == value
        ]
        if matches:
            resolved.append(matches[0])
            continue

        known = ", ".join(OLLAMA_MODEL_REGISTRY)
        raise ValueError(
            f"Unknown Ollama model {value!r}. Use a registry key ({known}), "
            "an exact model tag, or 'all'."
        )

    if not resolved:
        raise ValueError("--models must contain at least one model.")
    return resolved
