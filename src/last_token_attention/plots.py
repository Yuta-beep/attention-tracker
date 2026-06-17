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


def _ascii_safe(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _clean_token_label(token: str) -> str:
    token = token.replace("\n", "<NL>").replace("\t", "<TAB>")
    token = token.replace(" ", "<SP>")
    token = _ascii_safe(token)
    return token or "<EMPTY>"


def _token_tick_step(count: int) -> int:
    if count <= 28:
        return 1
    if count <= 60:
        return 2
    if count <= 100:
        return 4
    return max(1, count // 24)


def _token_tick_positions(tokens: list[str]) -> list[int]:
    step = _token_tick_step(len(tokens))
    return list(range(0, len(tokens), step))


def _token_labels(tokens: list[str], positions: list[int]) -> list[str]:
    labels = []
    for idx in positions:
        cleaned = _clean_token_label(tokens[idx])
        if len(cleaned) > 16:
            cleaned = cleaned[:16] + "..."
        labels.append(f"{idx}:{cleaned}")
    return labels


def _wrapped_prompt(prompt: str, width: int = 100) -> str:
    compact = prompt.replace("\n", "\\n")
    return "\n".join(textwrap.wrap(compact, width=width))


def _heatmap_panel(ax, matrix: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, xlabel: str, ylabel: str) -> None:
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, shrink=0.85)


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
        _heatmap_panel(ax, matrix, title, cmap, vmin, panel_vmax, "Head", "Layer")

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
            axes[0], normal, case_result["normal_token_texts"], case_result["instruction_token_indices_normal"], [],
            "Normal Prompt: Last Token Attention by Token Position", case_result["normal_prompt"],
        ),
        (
            axes[1], injected, case_result["injected_token_texts"], case_result["instruction_token_indices_injected"], case_result["injection_token_indices"],
            "Injected Prompt: Last Token Attention by Token Position", case_result["injected_prompt"],
        ),
    ]

    for ax, matrix, tokens, instruction_span, injection_span, title, prompt in panels:
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title, loc="left")
        ax.text(0.0, 1.02, _wrapped_prompt(prompt), transform=ax.transAxes, fontsize=8, ha="left", va="bottom", family="monospace")
        tick_positions = _token_tick_positions(tokens)
        ax.set_xlabel("Token Position / Token Fragment")
        ax.set_ylabel("Layer")
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            _token_labels(tokens, tick_positions),
            rotation=90,
            fontsize=7,
            family="monospace",
        )
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



def _mean_head_distraction(results: list[dict]) -> np.ndarray:
    matrices = []
    for row in results:
        normal = _as_matrix(row["normal_instruction_scores"])
        injected = _as_matrix(row["attack_instruction_scores"])
        matrices.append(normal - injected)
    return np.mean(np.stack(matrices, axis=0), axis=0)


def plot_mean_head_distraction(results: list[dict], output_path: str) -> None:
    matrix = _mean_head_distraction(results)
    vmax = float(np.abs(matrix).max()) or 1.0

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title("Mean Original-Instruction Attention Change Across Cases")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.text(
        0.0,
        -0.10,
        "Positive: attention to the original instruction decreased after injection (distraction).",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, label="Normal - Injected attention")

    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_distracted_heads(
    results: list[dict], output_path: str, top_k: int = 20
) -> None:
    matrix = _mean_head_distraction(results)
    flat = matrix.reshape(-1)
    k = min(top_k, flat.size)
    indices = np.argsort(flat)[-k:][::-1]
    num_heads = matrix.shape[1]
    labels = [f"L{index // num_heads}:H{index % num_heads}" for index in indices]
    values = flat[indices]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = np.arange(k)
    ax.bar(positions, values, color="#D45D00")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean normal - injected attention")
    ax.set_title(f"Top {k} Heads Most Distracted by Prompt Injection")
    ax.axhline(0.0, color="black", linewidth=0.8)

    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_dataset_delta_heatmaps(dataset_matrices: dict[str, np.ndarray], output_path: str) -> None:
    names = list(dataset_matrices)
    if not names:
        return
    vmax = max(float(np.abs(matrix).max()) for matrix in dataset_matrices.values()) or 1.0
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        _heatmap_panel(ax, dataset_matrices[name], f"{name}: mean(normal - attack)", "coolwarm", -vmax, vmax, "Head", "Layer")
    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmaps(matrices: dict[str, list[list[float]]], output_path: str) -> None:
    names = matrices["names"]
    panel_defs = [
        ("pearson", "Pearson"),
        ("spearman", "Spearman"),
        ("topk_overlap_ratio", "Top-k Overlap"),
        ("jaccard", "Jaccard"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panel_defs):
        matrix = np.asarray(matrices[key], dtype=float)
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0 if key in {"topk_overlap_ratio", "jaccard"} else -1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticklabels(names)
        plt.colorbar(im, ax=ax, shrink=0.85)
    path = Path(output_path)
    _ensure_parent(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
