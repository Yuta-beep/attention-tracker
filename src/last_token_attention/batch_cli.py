import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import MODEL_REGISTRY, resolve_model_config
from .runs import create_run_dir, slugify


DEFAULT_MODELS = [
    "qwen2_1.5b",
    "qwen2_7b",
    "llama3_8b",
    "mistral_7b",
    "phi3_mini",
    "gemma2_2b",
    "gemma2_9b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run last-token attention comparison for multiple models sequentially."
    )
    parser.add_argument("--input", default="", help="Input JSONL cases for compare_cli.")
    parser.add_argument(
        "--retry-manifest",
        default="",
        help="Retry only failed models from a previous manifest.json; reuses its input by default.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated registry keys/model ids, or 'all' for the default model set.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="all-models-last-token-attention")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument(
        "--detailed-plots",
        action="store_true",
        help="Also generate per-case plots for every model.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _split_models(raw_models: str) -> list[str]:
    if raw_models.strip().lower() == "all":
        return DEFAULT_MODELS.copy()
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must be 'all' or a non-empty comma-separated list.")
    return models


def _load_retry_manifest(path: str) -> tuple[list[str], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    models = [
        row["model_arg"]
        for row in payload.get("results", [])
        if row.get("status") == "failed"
    ]
    if not models:
        raise ValueError(f"No failed models found in retry manifest: {path}")
    input_path = payload.get("input", "")
    if not input_path:
        raise ValueError(f"Retry manifest does not contain an input path: {path}")
    return models, input_path


def _resolve_run_selection(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.retry_manifest:
        models, manifest_input = _load_retry_manifest(args.retry_manifest)
        return models, args.input or manifest_input
    if not args.input:
        raise ValueError("--input is required unless --retry-manifest is provided.")
    return _split_models(args.models), args.input


def _model_label(model: str) -> str:
    config = resolve_model_config(model)
    return slugify(config.key)


def _command_for_model(args: argparse.Namespace, parent_run_dir: Path, model: str) -> list[str]:
    label = _model_label(model)
    command = [
        sys.executable,
        "-m",
        "last_token_attention.compare_cli",
        "--model",
        model,
        "--input",
        args.input,
        "--output-root",
        str(parent_run_dir),
        "--experiment-title",
        label,
    ]
    if args.torch_dtype:
        command.extend(["--torch-dtype", args.torch_dtype])
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.no_chat_template:
        command.append("--no-chat-template")
    if args.no_system_role:
        command.append("--no-system-role")
    if args.detailed_plots:
        command.append("--detailed-plots")
    return command


def _parse_run_dir(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Run directory:"):
            return line.split(":", 1)[1].strip()
    return ""


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_model_comparison(manifest: dict, parent_run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for result in manifest["results"]:
        if result.get("status") != "success" or not result.get("run_dir"):
            continue
        metrics_path = Path(result["run_dir"]) / "plots" / "paper_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        focus = metrics["focus_k4"]
        rows.append({
            "model_key": result["model_key"],
            "model_id": result["model_id"],
            "auroc_k4": focus["auroc"],
            "num_important_heads_k4": focus["num_important_heads"],
            "head_proportion_k4": focus["head_proportion"],
        })
    if not rows:
        return

    output_dir = parent_run_dir / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    names = [row["model_key"] for row in rows]
    aucs = [row["auroc_k4"] for row in rows]
    proportions = [row["head_proportion_k4"] * 100 for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(x, aucs, color="#4C78A8")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Attention Tracker Detection by Model (k=4)")
    axes[1].bar(x, proportions, color="#F58518")
    axes[1].set_ylabel("Selected heads (%)")
    axes[1].set_title("Important Head Proportion by Model (k=4)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right")
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models, input_path = _resolve_run_selection(args)
    args.input = input_path
    parent_run_dir = create_run_dir(args.output_root, args.experiment_title)
    logs_dir = parent_run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_title": args.experiment_title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "models": models,
        "parent_run_dir": str(parent_run_dir),
        "results": [],
    }
    manifest_path = parent_run_dir / "manifest.json"

    for index, model in enumerate(models, start=1):
        config = resolve_model_config(
            model,
            torch_dtype=args.torch_dtype or None,
            load_in_4bit=True if args.load_in_4bit else None,
            uses_chat_template=False if args.no_chat_template else None,
            system_role_supported=False if args.no_system_role else None,
        )
        label = _model_label(model)
        command = _command_for_model(args, parent_run_dir, model)
        log_path = logs_dir / f"{index:02d}_{label}.log"
        command_text = shlex.join(command)

        row = {
            "model_arg": model,
            "model_key": config.key,
            "model_id": config.model_id,
            "status": "dry_run" if args.dry_run else "running",
            "returncode": None,
            "command": command_text,
            "log_path": str(log_path),
            "run_dir": "",
            "results_path": "",
        }
        manifest["results"].append(row)
        _write_manifest(manifest_path, manifest)

        print(f"[{index}/{len(models)}] {model}: {row['status']}", flush=True)
        if args.dry_run:
            log_path.write_text(command_text + "\n", encoding="utf-8")
            continue

        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        run_dir = _parse_run_dir(completed.stdout)
        row["returncode"] = completed.returncode
        row["run_dir"] = run_dir
        row["results_path"] = str(Path(run_dir) / "results.json") if run_dir else ""
        row["status"] = "success" if completed.returncode == 0 else "failed"
        _write_manifest(manifest_path, manifest)

        print(f"[{index}/{len(models)}] {model}: {row['status']}", flush=True)
        if completed.returncode != 0 and args.fail_fast:
            break

    successes = sum(1 for row in manifest["results"] if row["status"] == "success")
    failures = sum(1 for row in manifest["results"] if row["status"] == "failed")
    manifest["summary"] = {
        "num_models_requested": len(models),
        "num_success": successes,
        "num_failed": failures,
        "num_dry_run": sum(1 for row in manifest["results"] if row["status"] == "dry_run"),
    }
    _write_manifest(manifest_path, manifest)
    if not args.dry_run:
        _generate_model_comparison(manifest, parent_run_dir)

    readme = (
        "Generated files:\n"
        "- manifest.json: per-model commands, status, logs, and result paths\n"
        "- logs/*.log: stdout/stderr captured for each model run\n"
        "- */results.json: compare_cli output for each successful model\n"
        "- */plots/*.png: paper-style per-model analysis plots\n"
        "- model_comparison/: cross-model AUROC and important-head comparison\n"
    )
    (parent_run_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Run directory: {parent_run_dir.resolve()}")


if __name__ == "__main__":
    main()
