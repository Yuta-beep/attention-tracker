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


def _write_focus_outputs(results: list[dict], masks: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal = np.asarray([row["normal_instruction_scores"] for row in results], dtype=float)
    attack = np.asarray([row["attack_instruction_scores"] for row in results], dtype=float)

    focus_score_lines = []
    metrics = []
    for mask_row in masks:
        k = mask_row["k"]
        mask = mask_row["mask"]
        normal_focus = focus_scores_for_mask(normal, mask)
        attack_focus = focus_scores_for_mask(attack, mask)
        metrics.append(
            {
                "k": k,
                "num_important_heads": mask_row["num_important_heads"],
                "head_proportion": mask_row["head_proportion"],
                "auroc": _auroc(normal_focus, attack_focus),
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

    (output_dir / "focus_scores.jsonl").write_text(
        "\n".join(focus_score_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "focus_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


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

    first_matrix = np.asarray(results[0]["normal_instruction_scores"], dtype=float)
    masks = _load_head_masks(Path(args.head_selection_manifest), first_matrix.shape)
    focus_metrics = _write_focus_outputs(results, masks, focus_dir)

    payload = {
        "experiment_title": args.experiment_title,
        "run_dir": str(run_dir),
        "head_selection_manifest": args.head_selection_manifest,
        "model_key": config.key,
        "model_id": config.model_id,
        "summary": _build_summary(results),
        "focus_metrics_path": str(focus_dir / "focus_metrics.json"),
        "focus_scores_path": str(focus_dir / "focus_scores.jsonl"),
        "focus_metrics": focus_metrics,
        "cases": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- results.json: evaluation dataset attention outputs and focus metric pointers\n"
        "- focus_scores/focus_scores.jsonl: normal/attack focus scores for each case and k\n"
        "- focus_scores/focus_metrics.json: AUROC and selected-head counts for each k\n",
        encoding="utf-8",
    )

    print(f"Saved results: {output_path.resolve()}")
    print(f"Saved focus scores: {(focus_dir / 'focus_scores.jsonl').resolve()}")
    print(f"Saved focus metrics: {(focus_dir / 'focus_metrics.json').resolve()}")
    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
