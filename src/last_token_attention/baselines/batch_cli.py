import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from ..batch_cli import DEFAULT_MODELS, _split_models
from ..config import resolve_model_config
from ..runs import create_run_dir, slugify
from .cli import DETECTOR_NAMES


MODEL_DEPENDENT_DETECTORS = {"llm_based", "known_answer"}


@dataclass(frozen=True)
class BaselineJob:
    detector: str
    model_arg: str | None
    model_key: str | None
    model_id: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run trained detector baselines once and target-LLM baselines "
            "across all requested models."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--detectors",
        default="all",
        help="Comma-separated detectors or 'all'.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated target LLM registry keys/model ids or 'all'.",
    )
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="all-baseline-detectors")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_detectors(raw_detectors: str) -> list[str]:
    if raw_detectors.strip().lower() == "all":
        return list(DETECTOR_NAMES)
    detectors = [item.strip() for item in raw_detectors.split(",") if item.strip()]
    unknown = sorted(set(detectors) - set(DETECTOR_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown detectors: {', '.join(unknown)}. "
            f"Known detectors: {', '.join(DETECTOR_NAMES)}"
        )
    if not detectors:
        raise ValueError("--detectors must be 'all' or a non-empty comma-separated list.")
    return detectors


def build_jobs(detectors: list[str], models: list[str]) -> list[BaselineJob]:
    jobs = []
    for detector in detectors:
        if detector not in MODEL_DEPENDENT_DETECTORS:
            jobs.append(
                BaselineJob(
                    detector=detector,
                    model_arg=None,
                    model_key=None,
                    model_id="default",
                    label=detector,
                )
            )
            continue
        for model in models:
            config = resolve_model_config(model)
            jobs.append(
                BaselineJob(
                    detector=detector,
                    model_arg=model,
                    model_key=config.key,
                    model_id=config.model_id,
                    label=f"{detector}-{slugify(config.key)}",
                )
            )
    return jobs


def command_for_job(args: argparse.Namespace, parent_run_dir: Path, job: BaselineJob) -> list[str]:
    command = [
        sys.executable,
        "-m",
        f"last_token_attention.baselines.{job.detector}",
        "--input",
        args.input,
        "--output-root",
        str(parent_run_dir),
        "--experiment-title",
        job.label,
        "--separator",
        args.separator,
    ]
    if job.detector in MODEL_DEPENDENT_DETECTORS:
        command.extend(["--model", str(job.model_arg)])
        if args.torch_dtype:
            command.extend(["--torch-dtype", args.torch_dtype])
        if args.load_in_4bit:
            command.append("--load-in-4bit")
        if args.no_chat_template:
            command.append("--no-chat-template")
    else:
        command.extend(["--max-length", str(args.max_length)])
    return command


def parse_run_dir(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Run directory:"):
            return line.split(":", 1)[1].strip()
    return ""


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_comparison(manifest: dict, parent_run_dir: Path) -> None:
    rows = []
    for result in manifest["results"]:
        metrics_path = result.get("metrics_path")
        if result.get("status") != "success" or not metrics_path:
            continue
        path = Path(metrics_path)
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        summary = metrics["summary"]
        rows.append(
            {
                "detector": result["detector"],
                "model_key": result["model_key"],
                "model_id": result["model_id"],
                "micro_auroc": summary["micro_auroc"],
                "macro_auroc": summary.get("grouped", {}).get("macro_auroc"),
                "mean_latency_ms": summary["mean_latency_ms"],
                "metrics_path": metrics_path,
            }
        )
    if not rows:
        return
    comparison_dir = parent_run_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "baseline_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    columns = [
        "detector",
        "model_key",
        "model_id",
        "micro_auroc",
        "macro_auroc",
        "mean_latency_ms",
    ]
    csv_lines = [",".join(columns)]
    for row in rows:
        csv_lines.append(
            ",".join(
                "" if row[column] is None else str(row[column])
                for column in columns
            )
        )
    (comparison_dir / "baseline_comparison.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    detectors = split_detectors(args.detectors)
    models = DEFAULT_MODELS.copy() if args.models.strip().lower() == "all" else _split_models(args.models)
    jobs = build_jobs(detectors, models)

    parent_run_dir = create_run_dir(args.output_root, args.experiment_title)
    logs_dir = parent_run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parent_run_dir / "manifest.json"
    manifest = {
        "experiment_title": args.experiment_title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "detectors": detectors,
        "models": models,
        "num_planned_jobs": len(jobs),
        "parent_run_dir": str(parent_run_dir),
        "results": [],
    }

    for index, job in enumerate(jobs, start=1):
        command = command_for_job(args, parent_run_dir, job)
        command_text = shlex.join(command)
        log_path = logs_dir / f"{index:02d}_{job.label}.log"
        row = {
            "detector": job.detector,
            "model_arg": job.model_arg,
            "model_key": job.model_key,
            "model_id": job.model_id,
            "status": "dry_run" if args.dry_run else "running",
            "returncode": None,
            "command": command_text,
            "log_path": str(log_path),
            "run_dir": "",
            "metrics_path": "",
            "error": "",
        }
        manifest["results"].append(row)
        write_manifest(manifest_path, manifest)
        print(
            f"[{index}/{len(jobs)}] {job.detector}"
            f"{f'/{job.model_key}' if job.model_key else ''}: {row['status']}",
            flush=True,
        )

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
        run_dir = parse_run_dir(completed.stdout)
        row["returncode"] = completed.returncode
        row["run_dir"] = run_dir
        row["metrics_path"] = str(Path(run_dir) / "metrics.json") if run_dir else ""
        metrics_exists = bool(row["metrics_path"]) and Path(row["metrics_path"]).exists()
        row["status"] = (
            "success"
            if completed.returncode == 0 and run_dir and metrics_exists
            else "failed"
        )
        if row["status"] == "failed":
            if completed.returncode == 0 and not run_dir:
                row["error"] = (
                    "Child process exited successfully but did not report a run directory."
                )
            elif completed.returncode == 0 and not metrics_exists:
                row["error"] = (
                    f"Child process did not create expected metrics file: "
                    f"{row['metrics_path']}"
                )
            else:
                row["error"] = completed.stdout[-2000:]
        write_manifest(manifest_path, manifest)
        if completed.returncode != 0 and args.fail_fast:
            break

    manifest["summary"] = {
        "num_planned_jobs": len(jobs),
        "num_success": sum(row["status"] == "success" for row in manifest["results"]),
        "num_failed": sum(row["status"] == "failed" for row in manifest["results"]),
        "num_dry_run": sum(row["status"] == "dry_run" for row in manifest["results"]),
    }
    write_manifest(manifest_path, manifest)
    if not args.dry_run:
        generate_comparison(manifest, parent_run_dir)
    (parent_run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- manifest.json: all detector/model jobs and statuses\n"
        "- logs/*.log: captured output for every job\n"
        "- */predictions.jsonl and */metrics.json: per-job results\n"
        "- comparison/baseline_comparison.{json,csv}: cross-detector results\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Run directory: {parent_run_dir.resolve()}")


if __name__ == "__main__":
    main()
