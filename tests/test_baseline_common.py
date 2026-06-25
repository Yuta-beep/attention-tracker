import json

import pytest

from last_token_attention.baselines.common import (
    load_detection_examples,
    normalize_case,
)


def test_normalize_case_supports_base_text_format():
    normal, attack = normalize_case(
        {
            "id": "case-1",
            "instruction": "Classify.",
            "base_text": "clean",
            "injection_text": "ignore",
            "attack": "ignore",
        }
    )

    assert normal.text == "clean"
    assert normal.true_label == "normal"
    assert attack.text == "clean\n\nignore"
    assert attack.true_label == "attack"
    assert attack.metadata == {"attack": "ignore"}


def test_normalize_case_preserves_explicit_injected_text():
    _, attack = normalize_case(
        {
            "id": "case-2",
            "instruction": "Classify.",
            "normal_text": "clean",
            "injected_text": "custom attack text",
            "injection_text": "attack",
        }
    )

    assert attack.text == "custom attack text"


def test_load_detection_examples_reports_line_number(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "instruction": "Classify.",
                "base_text": "clean",
                "injection_text": "attack",
            }
        )
        + "\n"
        + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_detection_examples(path)
