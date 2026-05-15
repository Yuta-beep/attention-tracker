from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_DPI = 180


def _as_matrix(values: list[list[float]]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_token_label(token: str) -> str:
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    token = token.replace(" ", "␠")
    token = token.strip() or "␠"
    return token


def _token_labels(tokens: list[str]) -> list[str]:
    count = len(tokens)
    if count <= 28:
        step = 1
    elif count <= 60:
        step = 2
    elif count <= 100:
        step = 4
    else:
        step = max(1, count // 24)

    labels = []
    for idx, token in enumerate(tokens):
        if idx % step == 0:
            cleaned = _clean_token_label(token)
            if len(cleaned) > 10:
                cleaned = cleaned[:10] + "…"
            labels.append(f"{idx}:{cleaned}")
        else:
            labels.append("")
    return labels


def _wrapped_prompt(prompt: str, width: int = 100) -> str:
    compact = prompt.replace("\n", "\\n")
    return "\n".join(textwrap.wrap(compact, width=width))


def plot_case_heatmaps(case_result: dict, output_path: str) -> None:
    normal = _as_matrix(case_result["normal_instruction_scores"])
    attack_instruction = _as_matrix(case_result["attack_instruction_scores"])
    attack_injection = _as_matrix(case_result["attack_injection_scores"])
    delta = attack_instruction - normal

    vmax = float(max(normal.max(), attack_instruction.max(), attack_injection.max())) if normal.size else 1.0
    delta_abs = float(np.abs(delta).max()) if delta.size else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        (normal, "Normal -> Instruction", "viridis", 0.0, vmax),
        (attack_instruction, "Injected -> Instruction", "viridis", 0.0, vmax),
        (attack_injection, "Injected -> Injection", "viridis", 0.0, vmax),
        (delta, "Delta: Injected Instruction - Normal Instruction", "coolwarm", -delta_abs, delta_abs),
    ]

    for ax, (matrix, title, cmap, vmin, panel_vmax) in zip(axes.flat, panels):
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=panel_vmax)
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(case_result["id"], fontsize=14)
    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_case_token_position_heatmaps(case_result: dict, output_path: str) -> None:
    normal = _as_matrix(case_result["normal_token_scores"])
    injected = _as_matrix(case_result["injected_token_scores"])
    vmax = float(max(normal.max(), injected.max())) if normal.size else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), constrained_layout=True)
    panels = [
        (
            axes[0],
            normal,
            case_result["normal_token_texts"],
            case_result["instruction_token_indices_normal"],
            [],
            "Normal Prompt: Last Token Attention by Token Position",
            case_result["normal_prompt"],
        ),
        (
            axes[1],
            injected,
            case_result["injected_token_texts"],
            case_result["instruction_token_indices_injected"],
            case_result["injection_token_indices"],
            "Injected Prompt: Last Token Attention by Token Position",
            case_result["injected_prompt"],
        ),
    ]

    for ax, matrix, tokens, instruction_span, injection_span, title, prompt in panels:
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title, loc="left")
        ax.text(
            0.0,
            1.02,
            _wrapped_prompt(prompt),
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            family="monospace",
        )
        ax.set_xlabel("Token Position / Decoded Token")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(_token_labels(tokens), rotation=90, fontsize=7)
        if instruction_span:
            ax.axvspan(min(instruction_span) - 0.5, max(instruction_span) + 0.5, color="#4C78A8", alpha=0.18)
        if injection_span:
            ax.axvspan(min(injection_span) - 0.5, max(injection_span) + 0.5, color="#54A24B", alpha=0.18)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(case_result["id"], fontsize=14)
    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_case_totals(case_result: dict, output_path: str) -> None:
    labels = ["Normal -> Instruction", "Injected -> Instruction", "Injected -> Injection"]
    values = [
        case_result["normal_instruction_total"],
        case_result["attack_instruction_total"],
        case_result["attack_injection_total"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Total Attention Score")
    ax.set_title(case_result["id"])
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_summary_totals(results: list[dict], output_path: str) -> None:
    case_ids = [row["id"] for row in results]
    normal = np.asarray([row["normal_instruction_total"] for row in results], dtype=float)
    attack_instruction = np.asarray([row["attack_instruction_total"] for row in results], dtype=float)
    attack_injection = np.asarray([row["attack_injection_total"] for row in results], dtype=float)

    x = np.arange(len(case_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(case_ids) * 1.2), 5), constrained_layout=True)
    ax.bar(x - width, normal, width, label="Normal -> Instruction", color="#4C78A8")
    ax.bar(x, attack_instruction, width, label="Injected -> Instruction", color="#F58518")
    ax.bar(x + width, attack_injection, width, label="Injected -> Injection", color="#54A24B")

    ax.set_ylabel("Total Attention Score")
    ax.set_title("Last-Token Attention Comparison Across Cases")
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=30, ha="right")
    ax.legend()

    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
