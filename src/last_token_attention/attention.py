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

    per_span_layer_scores: dict[str, list[torch.Tensor]] = {name: [] for name in spans}
    per_layer_token_scores: list[torch.Tensor] = []
    span_indices = {
        name: torch.tensor(span.token_indices, device=device)
        for name, span in spans.items()
    }

    for layer_attn in outputs.attentions:
        last_token_to_all = layer_attn[0, :, -1, :]
        per_layer_token_scores.append(last_token_to_all.mean(dim=0))
        for name, key_indices in span_indices.items():
            head_scores = last_token_to_all.index_select(dim=-1, index=key_indices).sum(dim=-1)
            per_span_layer_scores[name].append(head_scores)

    per_layer_head_scores = {
        name: torch.stack(layer_scores, dim=0).detach().cpu()
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
        per_layer_token_scores=torch.stack(per_layer_token_scores, dim=0).detach().cpu(),
        span_token_indices=span_token_indices,
        per_layer_head_scores=per_layer_head_scores,
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
    return float(score_matrix.sum().item())
