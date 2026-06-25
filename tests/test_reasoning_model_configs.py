from last_token_attention.config import resolve_model_config


def test_reasoning_models_use_full_attention_layers():
    expected_loaders = {
        "qwen3.5_9b": "image_text_to_text",
        "gemma4_12b": "multimodal_lm",
        "gpt_oss_20b": "causal_lm",
    }

    for model_key, loader in expected_loaders.items():
        config = resolve_model_config(model_key)
        assert config.reasoning_enabled is True
        assert config.attn_implementation == "eager"
        assert config.attention_layer_type == "full_attention"
        assert config.model_loader == loader
