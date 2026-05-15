from dataclasses import dataclass


@dataclass
class TokenSpanResult:
    token_indices: list[int]
    char_start: int
    char_end: int


def _find_unique_substring_span(full_text: str, substring: str) -> tuple[int, int]:
    start = full_text.find(substring)
    if start < 0:
        raise ValueError("Instruction text was not found in the rendered prompt.")

    second = full_text.find(substring, start + 1)
    if second >= 0:
        raise ValueError(
            "Instruction text appears multiple times in the rendered prompt; unique span is required."
        )

    return start, start + len(substring)


def find_token_indices_for_substring(tokenizer, full_text: str, substring: str) -> TokenSpanResult:
    char_start, char_end = _find_unique_substring_span(full_text, substring)
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    offsets = encoded["offset_mapping"][0].tolist()
    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        overlaps = tok_start < char_end and tok_end > char_start
        if overlaps:
            token_indices.append(idx)

    if not token_indices:
        raise ValueError("No token indices overlap with the instruction span.")

    return TokenSpanResult(
        token_indices=token_indices,
        char_start=char_start,
        char_end=char_end,
    )
