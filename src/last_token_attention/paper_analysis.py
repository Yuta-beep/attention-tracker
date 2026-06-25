import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_DPI = 180
K_SWEEP_VALUES = [1.0 + 0.25 * index for index in range(13)]


def _matrices(results: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in results], dtype=float)


def _split_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.arange(0, count, 2)
    evaluation = np.arange(1, count, 2)
    if evaluation.size == 0:
        evaluation = calibration
    return calibration, evaluation


def _candidate_scores(normal: np.ndarray, attack: np.ndarray, k: float) -> np.ndarray:
    return (
        np.nanmean(normal, axis=0)
        - k * np.nanstd(normal, axis=0)
        - np.nanmean(attack, axis=0)
        - k * np.nanstd(attack, axis=0)
    )


def _important_mask(normal: np.ndarray, attack: np.ndarray, k: float) -> np.ndarray:
    return _candidate_scores(normal, attack, k) > 0.0


def focus_scores_for_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(values.shape[0], np.nan)
    return values[:, mask].mean(axis=1)


def _focus_scores(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return focus_scores_for_mask(values, mask)


def _auroc(normal_focus: np.ndarray, attack_focus: np.ndarray) -> float:
    normal_focus = normal_focus[np.isfinite(normal_focus)]
    attack_focus = attack_focus[np.isfinite(attack_focus)]
    if normal_focus.size == 0 or attack_focus.size == 0:
        return float("nan")
    comparisons = attack_focus[:, None] < normal_focus[None, :]
    ties = attack_focus[:, None] == normal_focus[None, :]
    return float((comparisons.sum() + 0.5 * ties.sum()) / comparisons.size)


def _roc_curve(normal_focus: np.ndarray, attack_focus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([np.zeros(len(normal_focus)), np.ones(len(attack_focus))])
    scores = -np.concatenate([normal_focus, attack_focus])
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if not len(scores):
        return np.asarray([]), np.asarray([])
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((labels == 0).sum()))
    tpr = []
    fpr = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(float(((predicted == 1) & (labels == 1)).sum() / positives))
        fpr.append(float(((predicted == 1) & (labels == 0)).sum() / negatives))
    return np.asarray(fpr), np.asarray(tpr)


def _attack_group(row: dict) -> str:
    text = row["injection_text"].lower()
    if "ignore" in text or "disregard" in text:
        return "ignore/disregard"
    if any(word in text for word in ["system", "developer", "policy", "safety"]):
        return "role/policy override"
    if any(word in text for word in ["---", "###", "new rule", "override:"]):
        return "delimiter/format"
    return "direct command"


def _ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _k_label(k: float) -> str:
    return f"{k:.2f}".replace(".", "_")


def _heads_from_mask(mask: np.ndarray, candidate_scores: np.ndarray) -> list[dict]:
    layers, heads = np.where(mask)
    return [
        {
            "layer": int(layer),
            "head": int(head),
            "candidate_score": float(candidate_scores[layer, head]),
        }
        for layer, head in zip(layers, heads)
    ]


def plot_figure2_head_maps(results: list[dict], path: Path) -> None:
    normal = np.nanmean(_matrices(results, "normal_instruction_scores"), axis=0)
    attack = np.nanmean(_matrices(results, "attack_instruction_scores"), axis=0)
    delta = normal - attack
    vmax = max(float(np.nanmax(normal)), float(np.nanmax(attack))) or 1.0
    dmax = float(np.nanmax(np.abs(delta))) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    panels = [
        (normal, "Normal: last token -> original instruction", "YlGnBu", 0.0, vmax),
        (attack, "Attack: last token -> original instruction", "YlGnBu", 0.0, vmax),
        (delta, "Distraction: normal - attack", "coolwarm", -dmax, dmax),
    ]
    for ax, (matrix, title, cmap, vmin, panel_max) in zip(axes, panels):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=panel_max)
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Paper-style Figure 2(a): Distraction Effect Across Heads")
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure2_token_shift(results: list[dict], path: Path) -> None:
    row = max(results, key=lambda item: -item["instruction_total_delta"])
    normal = np.asarray(row["normal_token_scores"], dtype=float)
    attack = np.asarray(row["injected_token_scores"], dtype=float)
    vmax = max(float(np.nanmax(normal)), float(np.nanmax(attack))) or 1.0
    fig, axes = plt.subplots(2, 1, figsize=(17, 9), constrained_layout=True)
    panels = [
        (axes[0], normal, row["normal_token_texts"], row["instruction_token_indices_normal"], [], "Normal"),
        (axes[1], attack, row["injected_token_texts"], row["instruction_token_indices_injected"], row["injection_token_indices"], "Attack"),
    ]
    for ax, matrix, tokens, instruction_span, injection_span, title in panels:
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Layer")
        ax.set_xlabel("Prompt token position")
        step = max(1, len(tokens) // 30)
        positions = list(range(0, len(tokens), step))
        labels = [tokens[index].replace("\n", "<NL>")[:12] for index in positions]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        if instruction_span:
            ax.axvspan(min(instruction_span) - 0.5, max(instruction_span) + 0.5, color="red", alpha=0.18, label="original instruction")
        if injection_span:
            ax.axvspan(min(injection_span) - 0.5, max(injection_span) + 0.5, color="lime", alpha=0.18, label="injected instruction")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Paper-style Figure 2(b): Token Attention Shift ({row['id']})")
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure3_distributions(results: list[dict], path: Path) -> None:
    groups: dict[str, list[float]] = {"no attack": []}
    for row in results:
        groups["no attack"].append(row["normal_instruction_total"])
        groups.setdefault(_attack_group(row), []).append(row["attack_instruction_total"])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, values in groups.items():
        values_array = np.asarray(values, dtype=float)
        bins = min(10, max(3, len(values_array)))
        ax.hist(values_array, bins=bins, density=True, histtype="step", linewidth=2, label=f"{name} (n={len(values)})")
    ax.set_title("Paper-style Figure 3: Original-Instruction Attention by Attack Strategy")
    ax.set_xlabel("Aggregated attention across all layers and heads")
    ax.set_ylabel("Density")
    ax.legend()
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure5_group_generalization(results: list[dict], path: Path) -> None:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in results:
        delta = np.asarray(row["normal_instruction_scores"], dtype=float) - np.asarray(row["attack_instruction_scores"], dtype=float)
        grouped.setdefault(_attack_group(row), []).append(delta)
    names = sorted(grouped)
    matrices = [np.nanmean(np.stack(grouped[name]), axis=0) for name in names]
    vmax = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices) or 1.0
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name, matrix in zip(axes, names, matrices):
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Paper-style Figure 5: Head Generalization Across Attack Styles")
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure8_candidate_scores(normal_cal: np.ndarray, attack_cal: np.ndarray, path: Path, k: float = 4.0) -> None:
    score = _candidate_scores(normal_cal, attack_cal, k)
    vmax = float(np.nanmax(np.abs(score))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    im = ax.imshow(score, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.contour(score > 0.0, levels=[0.5], colors="black", linewidths=0.5)
    ax.set_title(f"Paper-style Figure 8: Candidate Score by Head (k={k:g})")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label="candidate score")
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_focus_roc(normal_eval: np.ndarray, attack_eval: np.ndarray, normal_cal: np.ndarray, attack_cal: np.ndarray, path: Path, k: float = 4.0) -> dict:
    focus_metrics = calculate_focus_metrics(normal_eval, attack_eval, normal_cal, attack_cal, k)
    normal_focus = focus_metrics["normal_focus"]
    attack_focus = focus_metrics["attack_focus"]
    auc = focus_metrics["auroc"]
    fpr, tpr = _roc_curve(normal_focus, attack_focus)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    finite_normal = normal_focus[np.isfinite(normal_focus)]
    finite_attack = attack_focus[np.isfinite(attack_focus)]
    if finite_normal.size:
        axes[0].hist(finite_normal, bins=min(8, len(finite_normal)), alpha=0.65, density=True, label="normal")
    if finite_attack.size:
        axes[0].hist(finite_attack, bins=min(8, len(finite_attack)), alpha=0.65, density=True, label="attack")
    axes[0].set_title(f"Focus Score Distribution (k={k:g})")
    axes[0].set_xlabel("Focus score on important heads")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    if fpr.size:
        axes[1].plot(fpr, tpr, label=f"AUROC={auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Attention Tracker ROC")
    axes[1].legend()
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return {
        "k": k,
        "num_important_heads": focus_metrics["num_important_heads"],
        "head_proportion": focus_metrics["head_proportion"],
        "auroc": auc,
    }


def calculate_focus_metrics(normal_eval: np.ndarray, attack_eval: np.ndarray, normal_cal: np.ndarray, attack_cal: np.ndarray, k: float = 4.0) -> dict:
    mask = _important_mask(normal_cal, attack_cal, k)
    normal_focus = _focus_scores(normal_eval, mask)
    attack_focus = _focus_scores(attack_eval, mask)
    return {
        "k": k,
        "num_important_heads": int(mask.sum()),
        "head_proportion": float(mask.mean()),
        "auroc": _auroc(normal_focus, attack_focus),
        "normal_focus": normal_focus,
        "attack_focus": attack_focus,
    }


def calculate_k_ablation(normal_cal: np.ndarray, attack_cal: np.ndarray, normal_eval: np.ndarray, attack_eval: np.ndarray) -> list[dict]:
    rows = []
    for k in range(6):
        mask = _important_mask(normal_cal, attack_cal, float(k))
        auc = _auroc(_focus_scores(normal_eval, mask), _focus_scores(attack_eval, mask))
        rows.append({"k": k, "num_important_heads": int(mask.sum()), "head_proportion": float(mask.mean()), "auroc": auc})
    return rows


def plot_k_ablation(normal_cal: np.ndarray, attack_cal: np.ndarray, normal_eval: np.ndarray, attack_eval: np.ndarray, path: Path) -> list[dict]:
    rows = calculate_k_ablation(normal_cal, attack_cal, normal_eval, attack_eval)
    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ks = [row["k"] for row in rows]
    proportions = [row["head_proportion"] * 100 for row in rows]
    aucs = [row["auroc"] for row in rows]
    ax1.bar(ks, proportions, alpha=0.55, color="#4C78A8")
    ax1.set_xlabel("k in candidate score")
    ax1.set_ylabel("Selected heads (%)", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(ks, aucs, marker="o", color="#E45756")
    ax2.set_ylabel("Evaluation AUROC", color="#E45756")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Paper-style Table 2: Head Selection and Detection Performance")
    _ensure(path)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return rows


def generate_k_sweep(
    results: list[dict],
    normal_all: np.ndarray,
    attack_all: np.ndarray,
    normal_cal: np.ndarray,
    attack_cal: np.ndarray,
    normal_eval: np.ndarray,
    attack_eval: np.ndarray,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    focus_score_lines = []

    for k in K_SWEEP_VALUES:
        candidate_scores = _candidate_scores(normal_cal, attack_cal, k)
        mask = candidate_scores > 0.0
        selected_heads = _heads_from_mask(mask, candidate_scores)
        selected_heads_name = f"selected_heads_k{_k_label(k)}.json"
        (output_dir / selected_heads_name).write_text(
            json.dumps(
                {
                    "k": k,
                    "num_important_heads": len(selected_heads),
                    "head_proportion": float(mask.mean()),
                    "heads": selected_heads,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        plot_figure8_candidate_scores(
            normal_cal,
            attack_cal,
            output_dir / f"candidate_scores_k{_k_label(k)}.png",
            k=k,
        )
        focus_metrics = calculate_focus_metrics(
            normal_eval,
            attack_eval,
            normal_cal,
            attack_cal,
            k=k,
        )
        normal_focus_all = focus_scores_for_mask(normal_all, mask)
        attack_focus_all = focus_scores_for_mask(attack_all, mask)
        rows.append(
            {
                "k": k,
                "num_important_heads": focus_metrics["num_important_heads"],
                "head_proportion": focus_metrics["head_proportion"],
                "auroc": focus_metrics["auroc"],
                "plot": f"candidate_scores_k{_k_label(k)}.png",
                "selected_heads": selected_heads_name,
            }
        )

        for index, result in enumerate(results):
            split = "calibration" if index % 2 == 0 else "evaluation"
            common = {
                "k": k,
                "k_label": f"{k:.2f}",
                "case_id": result["id"],
                "split": split,
                "num_important_heads": int(mask.sum()),
            }
            focus_score_lines.append(
                json.dumps(
                    {
                        **common,
                        "label": "normal",
                        "focus_score": float(normal_focus_all[index]),
                    }
                )
            )
            focus_score_lines.append(
                json.dumps(
                    {
                        **common,
                        "label": "attack",
                        "focus_score": float(attack_focus_all[index]),
                    }
                )
            )

    (output_dir / "focus_scores.jsonl").write_text(
        "\n".join(focus_score_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "k_sweep_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "head_selection_manifest.json").write_text(
        json.dumps(
            {
                "k_values": K_SWEEP_VALUES,
                "split_note": "Important heads are selected from even-indexed calibration pairs.",
                "metrics": rows,
                "focus_scores": "focus_scores.jsonl",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "K sweep for Figure 8 candidate scores.\n"
        "Each PNG shows candidate_score by layer/head for one k value.\n"
        "Black contours indicate candidate_score > 0, i.e. selected important heads.\n"
        "k_sweep_metrics.json records selected-head counts, proportions, and AUROC.\n"
        "selected_heads_k*.json records the selected layer/head indices for one k value.\n"
        "focus_scores.jsonl records normal/attack focus scores for every case and k.\n"
        "head_selection_manifest.json is the input for separate evaluation runs.\n",
        encoding="utf-8",
    )
    return rows


def generate_paper_analysis(results: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal = _matrices(results, "normal_instruction_scores")
    attack = _matrices(results, "attack_instruction_scores")
    calibration_indices, evaluation_indices = _split_indices(len(results))
    normal_cal = normal[calibration_indices]
    attack_cal = attack[calibration_indices]
    normal_eval = normal[evaluation_indices]
    attack_eval = attack[evaluation_indices]

    plot_figure2_head_maps(results, output_dir / "01_figure2a_head_distraction.png")
    plot_figure2_token_shift(results, output_dir / "02_figure2b_token_shift.png")
    plot_figure3_distributions(results, output_dir / "03_figure3_attack_distributions.png")
    plot_figure8_candidate_scores(normal_cal, attack_cal, output_dir / "04_figure8_candidate_scores_k4.png")
    k_sweep = generate_k_sweep(results, normal, attack, normal_cal, attack_cal, normal_eval, attack_eval, output_dir / "k_sweep")
    focus_metrics = calculate_focus_metrics(normal_eval, attack_eval, normal_cal, attack_cal)
    k_ablation = calculate_k_ablation(normal_cal, attack_cal, normal_eval, attack_eval)
    focus_metrics_for_json = {
        key: value
        for key, value in focus_metrics.items()
        if key not in {"normal_focus", "attack_focus"}
    }

    metrics = {
        "method_note": "Important heads are selected on even-indexed pairs and evaluated on odd-indexed pairs.",
        "num_cases": len(results),
        "num_calibration_pairs": int(len(calibration_indices)),
        "num_evaluation_pairs": int(len(evaluation_indices)),
        "focus_k4": focus_metrics_for_json,
        "k_ablation": k_ablation,
        "k_sweep": k_sweep,
        "attack_groups": {name: sum(_attack_group(row) == name for row in results) for name in sorted({_attack_group(row) for row in results})},
    }
    (output_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Paper-style analysis based on Attention Tracker (arXiv:2411.00348).\n"
        "01-02: Figure 2-style distraction effect.\n"
        "03: Figure 3-style attack distribution comparison.\n"
        "04: Figure 8-style candidate score and important-head positions.\n"
        "k_sweep/: Figure 8 candidate scores for k=1.00..4.00 in 0.25 increments.\n"
        "paper_metrics.json: focus score AUROC and k/head-selection ablation metrics.\n"
        "This prompt-variation dataset is smaller and differs from the paper benchmarks.\n",
        encoding="utf-8",
    )
    return metrics
