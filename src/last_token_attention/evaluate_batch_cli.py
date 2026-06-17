import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .batch_cli import DEFAULT_MODELS, _split_models
from .config import resolve_model_config
from .runs import create_run_dir, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate focus scores for multiple models using head-selection manifests from a head-finding batch run."
    )
    parser.add_argument("--head-run", required=True, help="Parent head-finding batch run directory.")
    parser.add_argument("--input", required=True, help="Evaluation JSONL.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated registry keys/model ids, or 'all' for the default model set.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="eval-focus-all-models")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _model_label(model: str) -> str:
    config = resolve_model_config(model)
    return slugify(config.key)


def _find_head_manifest(head_run: Path, model: str) -> Path:
    label = _model_label(model)
    matches = sorted(head_run.glob(f"*_{label}/plots/k_sweep/head_selection_manifest.json"))
    if not matches:
        matches = [
            path
            for path in sorted(head_run.glob("*/plots/k_sweep/head_selection_manifest.json"))
            if f"_{label}" in str(path.parent.parent.parent.name)
        ]
    if not matches:
        raise FileNotFoundError(
            f"No head_selection_manifest.json found for model={model!r} under {head_run}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple head_selection_manifest.json files found for model={model!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _command_for_model(args: argparse.Namespace, parent_run_dir: Path, model: str, manifest_path: Path) -> list[str]:
    label = _model_label(model)
    command = [
        sys.executable,
        "-m",
        "last_token_attention.evaluate_focus_cli",
        "--head-selection-manifest",
        str(manifest_path),
        "--input",
        args.input,
        "--model",
        model,
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
    return command


def _parse_run_dir(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Run directory:"):
            return line.split(":", 1)[1].strip()
    return ""


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_comparison(manifest: dict, parent_run_dir: Path) -> None:
    rows = []
    for result in manifest["results"]:
        if result.get("status") != "success" or not result.get("run_dir"):
            continue
        metrics_path = Path(result["run_dir"]) / "focus_scores" / "focus_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_key": result["model_key"],
                "model_id": result["model_id"],
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )
    if not rows:
        return

    output_dir = parent_run_dir / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_model_comparison.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    models = DEFAULT_MODELS.copy() if args.models.strip().lower() == "all" else _split_models(args.models)
    head_run = Path(args.head_run)
    if not head_run.exists():
        raise FileNotFoundError(f"--head-run not found: {head_run}")

    parent_run_dir = create_run_dir(args.output_root, args.experiment_title)
    logs_dir = parent_run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parent_run_dir / "manifest.json"

    manifest = {
        "experiment_title": args.experiment_title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "head_run": str(head_run),
        "models": models,
        "parent_run_dir": str(parent_run_dir),
        "results": [],
    }

    for index, model in enumerate(models, start=1):
        config = resolve_model_config(
            model,
            torch_dtype=args.torch_dtype or None,
            load_in_4bit=True if args.load_in_4bit else None,
            uses_chat_template=False if args.no_chat_template else None,
            system_role_supported=False if args.no_system_role else None,
        )
        label = _model_label(model)
        try:
            head_manifest = _find_head_manifest(head_run, model)
        except Exception as exc:
            row = {
                "model_arg": model,
                "model_key": config.key,
                "model_id": config.model_id,
                "status": "failed",
                "returncode": None,
                "error": str(exc),
                "command": "",
                "log_path": "",
                "head_selection_manifest": "",
                "run_dir": "",
                "focus_metrics_path": "",
            }
            manifest["results"].append(row)
            _write_manifest(manifest_path, manifest)
            print(f"[{index}/{len(models)}] {model}: failed ({exc})", flush=True)
            if args.fail_fast:
                break
            continue

        command = _command_for_model(args, parent_run_dir, model, head_manifest)
        log_path = logs_dir / f"{index:02d}_{label}.log"
        row = {
            "model_arg": model,
            "model_key": config.key,
            "model_id": config.model_id,
            "status": "dry_run" if args.dry_run else "running",
            "returncode": None,
            "error": "",
            "command": shlex.join(command),
            "log_path": str(log_path),
            "head_selection_manifest": str(head_manifest),
            "run_dir": "",
            "focus_metrics_path": "",
        }
        manifest["results"].append(row)
        _write_manifest(manifest_path, manifest)

        print(f"[{index}/{len(models)}] {model}: {row['status']}", flush=True)
        if args.dry_run:
            log_path.write_text(row["command"] + "\n", encoding="utf-8")
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
        row["focus_metrics_path"] = str(Path(run_dir) / "focus_scores" / "focus_metrics.json") if run_dir else ""
        row["status"] = "success" if completed.returncode == 0 else "failed"
        _write_manifest(manifest_path, manifest)

        print(f"[{index}/{len(models)}] {model}: {row['status']}", flush=True)
        if completed.returncode != 0 and args.fail_fast:
            break

    manifest["summary"] = {
        "num_models_requested": len(models),
        "num_success": sum(1 for row in manifest["results"] if row["status"] == "success"),
        "num_failed": sum(1 for row in manifest["results"] if row["status"] == "failed"),
        "num_dry_run": sum(1 for row in manifest["results"] if row["status"] == "dry_run"),
    }
    _write_manifest(manifest_path, manifest)
    if not args.dry_run:
        _generate_comparison(manifest, parent_run_dir)

    (parent_run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- manifest.json: per-model evaluation command, status, logs, and result paths\n"
        "- logs/*.log: stdout/stderr captured for each model run\n"
        "- */results.json: evaluate_focus_cli output for each successful model\n"
        "- */focus_scores/focus_metrics.json: per-k AUROC and selected-head counts\n"
        "- model_comparison/focus_model_comparison.json: cross-model focus metrics\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Run directory: {parent_run_dir.resolve()}")


if __name__ == "__main__":
    main()
