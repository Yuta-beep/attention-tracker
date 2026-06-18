import argparse
import json
from pathlib import Path

import numpy as np

from .compare_cli import _analyze_case, _build_summary, _load_cases, _normalize_case
from .config import resolve_model_config
from .modeling import load_model_bundle
from .paper_analysis import _auroc, focus_scores_for_mask
from .runs import create_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate focus scores on a separate dataset using selected heads from a head-finding run."
    )
    parser.add_argument("--head-selection-manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="qwen2_1.5b")
    parser.add_argument("--model-id", default="", help="Explicit Hugging Face model id; overrides --model.")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="focus-score-evaluation")
    parser.add_argument(
        "--threshold-strategy",
        choices=["youden", "max-accuracy"],
        default="youden",
        help="How to choose the focus-score threshold from the head-finding calibration scores.",
    )
    return parser.parse_args()


def _load_head_masks(manifest_path: Path, matrix_shape: tuple[int, int]) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    masks = []
    for row in manifest["metrics"]:
        selected_heads_path = base_dir / row["selected_heads"]
        selected_heads_payload = json.loads(selected_heads_path.read_text(encoding="utf-8"))
        mask = np.zeros(matrix_shape, dtype=bool)
        for head in selected_heads_payload["heads"]:
            layer = int(head["layer"])
            head_index = int(head["head"])
            if layer >= matrix_shape[0] or head_index >= matrix_shape[1]:
                raise ValueError(
                    f"Selected head L{layer}:H{head_index} does not fit evaluation matrix shape {matrix_shape}."
                )
            mask[layer, head_index] = True
        masks.append(
            {
                "k": float(row["k"]),
                "num_important_heads": int(mask.sum()),
                "head_proportion": float(mask.mean()),
                "mask": mask,
                "source": str(selected_heads_path),
            }
        )
    return masks


def _load_calibration_focus_scores(manifest_path: Path) -> dict[float, dict[str, np.ndarray]]:
    focus_scores_path = manifest_path.parent / "focus_scores.jsonl"
    if not focus_scores_path.exists():
        return {}

    grouped: dict[float, dict[str, list[float]]] = {}
    fallback_grouped: dict[float, dict[str, list[float]]] = {}
    for line in focus_scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        k = float(row["k"])
        label = row["label"]
        if label not in {"normal", "attack"}:
            continue
        fallback_grouped.setdefault(k, {"normal": [], "attack": []})
        fallback_grouped[k][label].append(float(row["focus_score"]))
        if row.get("split", "calibration") != "calibration":
            continue
        grouped.setdefault(k, {"normal": [], "attack": []})
        grouped[k][label].append(float(row["focus_score"]))

    if not grouped:
        grouped = fallback_grouped

    return {
        k: {
            label: np.asarray(values, dtype=float)
            for label, values in labels.items()
        }
        for k, labels in grouped.items()
    }


def _confusion_metrics(
    normal_focus: np.ndarray,
    attack_focus: np.ndarray,
    threshold: float,
) -> dict:
    normal_focus = normal_focus[np.isfinite(normal_focus)]
    attack_focus = attack_focus[np.isfinite(attack_focus)]
    normal_pred_attack = normal_focus < threshold
    attack_pred_attack = attack_focus < threshold

    fp = int(normal_pred_attack.sum())
    tn = int((~normal_pred_attack).sum())
    tp = int(attack_pred_attack.sum())
    fn = int((~attack_pred_attack).sum())
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    accuracy = (tp + tn) / total if total else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    youden_j = recall - fpr if np.isfinite(recall) and np.isfinite(fpr) else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "f1": float(f1),
        "youden_j": float(youden_j),
    }


def _candidate_thresholds(normal_focus: np.ndarray, attack_focus: np.ndarray) -> np.ndarray:
    values = np.concatenate([normal_focus, attack_focus])
    values = np.sort(np.unique(values[np.isfinite(values)]))
    if values.size == 0:
        return np.asarray([])
    if values.size == 1:
        return values
    return np.concatenate([
        [values[0] - 1e-12],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + 1e-12],
    ])


def _select_threshold(
    normal_focus: np.ndarray,
    attack_focus: np.ndarray,
    strategy: str,
) -> dict:
    thresholds = _candidate_thresholds(normal_focus, attack_focus)
    if thresholds.size == 0:
        return {
            "threshold": None,
            "strategy": strategy,
            "calibration_metrics": None,
        }

    scored = [_confusion_metrics(normal_focus, attack_focus, threshold) for threshold in thresholds]
    if strategy == "max-accuracy":
        best = max(
            scored,
            key=lambda row: (
                np.nan_to_num(row["accuracy"], nan=-1.0),
                np.nan_to_num(row["recall"], nan=-1.0),
                -row["threshold"],
            ),
        )
    else:
        best = max(
            scored,
            key=lambda row: (
                np.nan_to_num(row["youden_j"], nan=-1.0),
                np.nan_to_num(row["accuracy"], nan=-1.0),
                -row["threshold"],
            ),
        )
    return {
        "threshold": best["threshold"],
        "strategy": strategy,
        "calibration_metrics": best,
    }


def _apply_threshold(focus_score: float, threshold: float | None) -> dict:
    if threshold is None or not np.isfinite(focus_score):
        return {
            "predicted_label": "unknown",
            "is_injection": None,
            "decision": "unknown",
        }
    is_injection = bool(focus_score < threshold)
    return {
        "predicted_label": "attack" if is_injection else "normal",
        "is_injection": is_injection,
        "decision": "reject" if is_injection else "accept",
    }


def _write_focus_outputs(
    results: list[dict],
    masks: list[dict],
    output_dir: Path,
    calibration_scores: dict[float, dict[str, np.ndarray]],
    threshold_strategy: str,
) -> tuple[list[dict], list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal = np.asarray([row["normal_instruction_scores"] for row in results], dtype=float)
    attack = np.asarray([row["attack_instruction_scores"] for row in results], dtype=float)

    focus_score_lines = []
    detection_lines = []
    metrics = []
    for mask_row in masks:
        k = mask_row["k"]
        mask = mask_row["mask"]
        normal_focus = focus_scores_for_mask(normal, mask)
        attack_focus = focus_scores_for_mask(attack, mask)
        threshold_payload = _select_threshold(
            calibration_scores.get(k, {}).get("normal", np.asarray([], dtype=float)),
            calibration_scores.get(k, {}).get("attack", np.asarray([], dtype=float)),
            threshold_strategy,
        )
        threshold = threshold_payload["threshold"]
        evaluation_metrics = (
            _confusion_metrics(normal_focus, attack_focus, threshold)
            if threshold is not None
            else None
        )
        metrics.append(
            {
                "k": k,
                "num_important_heads": mask_row["num_important_heads"],
                "head_proportion": mask_row["head_proportion"],
                "auroc": _auroc(normal_focus, attack_focus),
                "threshold": threshold,
                "threshold_strategy": threshold_strategy,
                "threshold_source": "head_selection_focus_scores",
                "calibration_detection_metrics": threshold_payload["calibration_metrics"],
                "evaluation_detection_metrics": evaluation_metrics,
                "selected_heads_source": mask_row["source"],
            }
        )
        for index, result in enumerate(results):
            common = {
                "k": k,
                "k_label": f"{k:.2f}",
                "case_id": result["id"],
                "num_important_heads": mask_row["num_important_heads"],
            }
            focus_score_lines.append(
                json.dumps(
                    {
                        **common,
                        "label": "normal",
                        "focus_score": float(normal_focus[index]),
                    }
                )
            )
            focus_score_lines.append(
                json.dumps(
                    {
                        **common,
                        "label": "attack",
                        "focus_score": float(attack_focus[index]),
                    }
                )
            )
            for label, focus_score in (
                ("normal", float(normal_focus[index])),
                ("attack", float(attack_focus[index])),
            ):
                prediction = _apply_threshold(focus_score, threshold)
                detection_lines.append(
                    json.dumps(
                        {
                            **common,
                            "true_label": label,
                            "focus_score": focus_score,
                            "threshold": threshold,
                            **prediction,
                        }
                    )
                )

    (output_dir / "focus_scores.jsonl").write_text(
        "\n".join(focus_score_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detections.jsonl").write_text(
        "\n".join(detection_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "focus_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics, detection_lines


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.output_root, args.experiment_title)
    output_path = run_dir / "results.json"
    focus_dir = run_dir / "focus_scores"

    config = resolve_model_config(
        args.model_id or args.model,
        torch_dtype=args.torch_dtype or None,
        load_in_4bit=True if args.load_in_4bit else None,
        uses_chat_template=False if args.no_chat_template else None,
        system_role_supported=False if args.no_system_role else None,
    )
    bundle = load_model_bundle(config)
    raw_cases = _load_cases(args)
    cases = [_normalize_case(case, args.separator) for case in raw_cases]
    results = [_analyze_case(bundle, config, case) for case in cases]

    head_selection_manifest = Path(args.head_selection_manifest)
    first_matrix = np.asarray(results[0]["normal_instruction_scores"], dtype=float)
    masks = _load_head_masks(head_selection_manifest, first_matrix.shape)
    calibration_scores = _load_calibration_focus_scores(head_selection_manifest)
    focus_metrics, _ = _write_focus_outputs(
        results,
        masks,
        focus_dir,
        calibration_scores,
        args.threshold_strategy,
    )

    payload = {
        "experiment_title": args.experiment_title,
        "run_dir": str(run_dir),
        "head_selection_manifest": args.head_selection_manifest,
        "threshold_strategy": args.threshold_strategy,
        "model_key": config.key,
        "model_id": config.model_id,
        "summary": _build_summary(results),
        "focus_metrics_path": str(focus_dir / "focus_metrics.json"),
        "focus_scores_path": str(focus_dir / "focus_scores.jsonl"),
        "detections_path": str(focus_dir / "detections.jsonl"),
        "focus_metrics": focus_metrics,
        "cases": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- results.json: evaluation dataset attention outputs and focus metric pointers\n"
        "- focus_scores/focus_scores.jsonl: normal/attack focus scores for each case and k\n"
        "- focus_scores/detections.jsonl: thresholded accept/reject decisions for each case and k\n"
        "- focus_scores/focus_metrics.json: AUROC, selected-head counts, thresholds, and detection metrics for each k\n",
        encoding="utf-8",
    )

    print(f"Saved results: {output_path.resolve()}")
    print(f"Saved focus scores: {(focus_dir / 'focus_scores.jsonl').resolve()}")
    print(f"Saved detections: {(focus_dir / 'detections.jsonl').resolve()}")
    print(f"Saved focus metrics: {(focus_dir / 'focus_metrics.json').resolve()}")
    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
