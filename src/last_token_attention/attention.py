from dataclasses import dataclass

import torch

from .token_span import TokenSpanResult


@dataclass
class AttentionExtractionResult:
    prompt: str
    input_ids: torch.Tensor
    instruction_token_indices: list[int]
    per_layer_head_scores: torch.Tensor


@dataclass
class MultiSpanAttentionResult:
    prompt: str
    input_ids: torch.Tensor
    token_texts: list[str]
    per_layer_token_scores: torch.Tensor
    span_token_indices: dict[str, list[int]]
    per_layer_head_scores: dict[str, torch.Tensor]
    active_layer_indices: list[int]


@dataclass
class GeneratedReasoningAttentionResult:
    prompt: str
    generated_text: str
    generated_token_ids: list[int]
    generated_token_texts: list[str]
    active_layer_indices: list[int]
    per_token_layer_head_scores: dict[str, torch.Tensor]


def _layer_types(model) -> list[str] | None:
    config = model.config
    text_config = getattr(config, "text_config", None)
    return getattr(text_config, "layer_types", None) or getattr(config, "layer_types", None)


def _is_active_attention_layer(model_bundle, layer_index: int) -> bool:
    required_type = model_bundle.config.attention_layer_type
    if not required_type:
        return True
    layer_types = _layer_types(model_bundle.model)
    if not layer_types or layer_index >= len(layer_types):
        return True
    return layer_types[layer_index] == required_type


@torch.inference_mode()
def extract_last_token_attention_to_spans(
    model_bundle,
    prompt: str,
    spans: dict[str, TokenSpanResult],
) -> MultiSpanAttentionResult:
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = model_bundle.device

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model(**encoded, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions. Check output_attentions=True.")

    first_attention = next(
        (
            attention
            for index, attention in enumerate(outputs.attentions)
            if attention is not None and _is_active_attention_layer(model_bundle, index)
        ),
        None,
    )
    if first_attention is None:
        raise RuntimeError(
            "Model returned no usable attention tensors. "
            "Try attn_implementation='eager' or check attention_layer_type."
        )
    num_layers = len(outputs.attentions)
    num_heads = first_attention.shape[1]
    sequence_length = encoded["input_ids"].shape[1]
    per_span_layer_scores = {
        name: torch.full(
            (num_layers, num_heads),
            torch.nan,
            dtype=first_attention.dtype,
            device=device,
        )
        for name in spans
    }
    per_layer_token_scores = torch.full(
        (num_layers, sequence_length),
        torch.nan,
        dtype=first_attention.dtype,
        device=device,
    )
    active_layer_indices = []
    span_indices = {
        name: torch.tensor(span.token_indices, device=device)
        for name, span in spans.items()
    }

    for layer_index, layer_attn in enumerate(outputs.attentions):
        if layer_attn is None or not _is_active_attention_layer(model_bundle, layer_index):
            continue
        last_token_to_all = layer_attn[0, :, -1, :]
        if last_token_to_all.shape[-1] != sequence_length:
            continue
        active_layer_indices.append(layer_index)
        per_layer_token_scores[layer_index] = last_token_to_all.mean(dim=0)
        for name, key_indices in span_indices.items():
            head_scores = last_token_to_all.index_select(dim=-1, index=key_indices).sum(dim=-1)
            per_span_layer_scores[name][layer_index] = head_scores

    per_layer_head_scores = {
        name: layer_scores.detach().cpu()
        for name, layer_scores in per_span_layer_scores.items()
    }
    span_token_indices = {name: span.token_indices for name, span in spans.items()}
    input_ids_cpu = encoded["input_ids"].detach().cpu()
    token_ids = input_ids_cpu[0].tolist()
    token_texts = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]
    return MultiSpanAttentionResult(
        prompt=prompt,
        input_ids=input_ids_cpu,
        token_texts=token_texts,
        per_layer_token_scores=per_layer_token_scores.detach().cpu(),
        span_token_indices=span_token_indices,
        per_layer_head_scores=per_layer_head_scores,
        active_layer_indices=active_layer_indices,
    )


@torch.inference_mode()
def extract_generated_reasoning_attention_to_spans(
    model_bundle,
    prompt: str,
    spans: dict[str, TokenSpanResult],
    max_new_tokens: int = 256,
) -> GeneratedReasoningAttentionResult:
    tokenizer = model_bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_bundle.device)
    prompt_length = input_ids.shape[1]
    generation = model_bundle.model.generate(
        input_ids=input_ids,
        attention_mask=encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            model_bundle.device
        ),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = generation.sequences[0, prompt_length:]
    active_layers = [
        index
        for index in range(len(generation.attentions[0]))
        if _is_active_attention_layer(model_bundle, index)
    ] if generation.attentions else []
    if not active_layers:
        raise RuntimeError("Generation returned no usable reasoning attention tensors.")

    first = next(
        generation.attentions[step][layer]
        for step in range(len(generation.attentions))
        for layer in active_layers
        if generation.attentions[step][layer] is not None
    )
    num_layers = len(generation.attentions[0])
    num_heads = first.shape[1]
    scores = {
        name: torch.full(
            (len(generation.attentions), num_layers, num_heads),
            torch.nan,
            dtype=first.dtype,
        )
        for name in spans
    }
    for step_index, step_attentions in enumerate(generation.attentions):
        for layer_index in active_layers:
            layer_attention = step_attentions[layer_index]
            if layer_attention is None:
                continue
            last_query = layer_attention[0, :, -1, :]
            for name, span in spans.items():
                valid_indices = [
                    index for index in span.token_indices if index < last_query.shape[-1]
                ]
                if not valid_indices:
                    continue
                key_indices = torch.tensor(valid_indices, device=last_query.device)
                scores[name][step_index, layer_index] = (
                    last_query.index_select(-1, key_indices).sum(-1).detach().cpu()
                )

    token_ids = generated_ids.detach().cpu().tolist()
    return GeneratedReasoningAttentionResult(
        prompt=prompt,
        generated_text=tokenizer.decode(generated_ids, skip_special_tokens=False),
        generated_token_ids=token_ids,
        generated_token_texts=[
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ],
        active_layer_indices=active_layers,
        per_token_layer_head_scores=scores,
    )


@torch.inference_mode()
def extract_last_token_instruction_attention(
    model_bundle,
    prompt: str,
    instruction_span: TokenSpanResult,
) -> AttentionExtractionResult:
    result = extract_last_token_attention_to_spans(
        model_bundle,
        prompt,
        {"instruction": instruction_span},
    )
    return AttentionExtractionResult(
        prompt=result.prompt,
        input_ids=result.input_ids,
        instruction_token_indices=result.span_token_indices["instruction"],
        per_layer_head_scores=result.per_layer_head_scores["instruction"],
    )


def sum_attention_scores(score_matrix: torch.Tensor) -> float:
    return float(torch.nansum(score_matrix).item())
