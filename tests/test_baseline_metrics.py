import pytest

from last_token_attention.baselines.metrics import auroc, summarize


def _row(label, score, case_id, **metadata):
    return {
        "case_id": case_id,
        "true_label": label,
        "attack_score": score,
        "latency_ms": 1.0,
        "metadata": metadata,
    }


def test_auroc_uses_higher_score_as_attack():
    rows = [
        _row("normal", 0.1, "a"),
        _row("normal", 0.2, "b"),
        _row("attack", 0.8, "a"),
        _row("attack", 0.9, "b"),
    ]

    assert auroc(rows) == 1.0


def test_auroc_counts_ties_as_half():
    rows = [
        _row("normal", 0.5, "a"),
        _row("attack", 0.5, "a"),
    ]

    assert auroc(rows) == 0.5


def test_summary_includes_grouped_macro_auroc():
    rows = [
        _row("normal", 0.1, "a", attack="ignore"),
        _row("attack", 0.9, "a", attack="ignore"),
        _row("normal", 0.8, "b", attack="naive"),
        _row("attack", 0.2, "b", attack="naive"),
    ]

    summary = summarize(rows)

    assert summary["micro_auroc"] == pytest.approx(0.75)
    assert summary["grouped"]["macro_auroc"] == pytest.approx(0.5)
