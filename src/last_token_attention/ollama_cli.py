import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from .ollama_models import OLLAMA_MODEL_REGISTRY, OllamaModelConfig, resolve_ollama_models
from .runs import create_run_dir


DEFAULT_PROMPT = (
    "Explain how transformer attention helps a language model use context. "
    "Answer in one concise paragraph."
)
DEFAULT_INPUT = "data.head_finding.practical_30.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage and benchmark the recommended local Ollama inference models."
    )
    parser.add_argument("action", choices=["list", "status", "pull", "benchmark"])
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated registry keys/model tags, or 'all'.",
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Attention Tracker JSONL cases. Each case is run as a normal/attack "
            "pair. Pass an empty string to use --prompt instead."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of JSONL cases for a smoke test.",
    )
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=0,
        help="Override the per-model context recommendation.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="ollama-inference-benchmark")
    return parser.parse_args()


def _request_json(
    host: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 600,
) -> dict:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc


def _installed_models(host: str) -> set[str]:
    payload = _request_json(host, "/api/tags", timeout=30)
    return {
        row.get("name", "")
        for row in payload.get("models", [])
        if row.get("name")
    }


def _print_registry() -> None:
    for config in OLLAMA_MODEL_REGISTRY.values():
        print(
            f"{config.key:14} {config.model:16} "
            f"context={config.default_context:6d} "
            f"download~{config.expected_download_gb:4.1f}GB  {config.role}"
        )


def _print_status(host: str) -> None:
    installed = _installed_models(host)
    for config in OLLAMA_MODEL_REGISTRY.values():
        state = "installed" if config.model in installed else "missing"
        print(f"{config.model:16} {state}")


def _pull_models(models: list[OllamaModelConfig]) -> None:
    for index, config in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] Pulling {config.model}", flush=True)
        completed = subprocess.run(
            ["ollama", "pull", config.model],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ollama pull failed for {config.model} "
                f"with exit code {completed.returncode}"
            )


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return float(count / (duration_ns / 1_000_000_000))


def _load_cases(input_path: str, limit: int = 0) -> list[dict]:
    cases = []
    for line_number, line in enumerate(
        Path(input_path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {
            key
            for key in ("instruction", "normal_text", "injected_text")
            if key not in row
        }
        if missing:
            raise ValueError(
                f"{input_path}:{line_number} is missing fields: "
                + ", ".join(sorted(missing))
            )
        cases.append(row)
        if limit and len(cases) >= limit:
            break
    if not cases:
        raise ValueError(f"No cases found in {input_path}.")
    return cases


def _response_metrics(response: dict) -> dict:
    message = response.get("message", {})
    return {
        "wall_seconds": response["_wall_seconds"],
        "load_seconds": response.get("load_duration", 0) / 1_000_000_000,
        "prompt_tokens": response.get("prompt_eval_count"),
        "prompt_tokens_per_second": _tokens_per_second(
            response.get("prompt_eval_count"),
            response.get("prompt_eval_duration"),
        ),
        "generated_tokens": response.get("eval_count"),
        "generation_tokens_per_second": _tokens_per_second(
            response.get("eval_count"),
            response.get("eval_duration"),
        ),
        "done_reason": response.get("done_reason"),
        "response": message.get("content", response.get("response", "")),
        "thinking": message.get("thinking", response.get("thinking", "")),
    }


def _run_chat(
    host: str,
    config: OllamaModelConfig,
    args: argparse.Namespace,
    instruction: str,
    user_text: str,
) -> dict:
    context = args.num_ctx or config.default_context
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "think": args.think,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "num_ctx": context,
        },
    }
    started = time.perf_counter()
    response = _request_json(host, "/api/chat", payload=payload)
    response["_wall_seconds"] = time.perf_counter() - started
    return {
        "model_key": config.key,
        "model": config.model,
        "role": config.role,
        "context": context,
        "thinking_enabled": args.think,
        **_response_metrics(response),
    }


def _mean_finite(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _summarize_results(results: list[dict]) -> list[dict]:
    summaries = []
    model_names = list(dict.fromkeys(row["model"] for row in results))
    for model in model_names:
        model_rows = [row for row in results if row["model"] == model]
        summaries.append(
            {
                "model": model,
                "num_inferences": len(model_rows),
                "mean_wall_seconds": _mean_finite(model_rows, "wall_seconds"),
                "mean_prompt_tokens_per_second": _mean_finite(
                    model_rows, "prompt_tokens_per_second"
                ),
                "mean_generation_tokens_per_second": _mean_finite(
                    model_rows, "generation_tokens_per_second"
                ),
            }
        )
    return summaries


def _run_benchmark(
    host: str,
    models: list[OllamaModelConfig],
    args: argparse.Namespace,
) -> Path:
    installed = _installed_models(host)
    missing = [config.model for config in models if config.model not in installed]
    if missing:
        raise RuntimeError(
            "Models are not installed: "
            + ", ".join(missing)
            + ". Run `manage-ollama-models pull --models all` first."
        )

    cases = _load_cases(args.input, args.limit) if args.input else []
    if not cases:
        cases = [
            {
                "id": "single-prompt",
                "instruction": "Follow the user's request.",
                "normal_text": args.prompt,
                "injected_text": args.prompt,
            }
        ]

    run_dir = create_run_dir(args.output_root, args.experiment_title)
    predictions_path = run_dir / "ollama_predictions.jsonl"
    results = []
    total = len(models) * len(cases) * 2
    completed = 0
    for config in models:
        for case in cases:
            for label, field in (("normal", "normal_text"), ("attack", "injected_text")):
                completed += 1
                case_id = case.get("id", f"case-{completed:04d}")
                print(
                    f"[{completed}/{total}] {config.model} {case_id} {label}",
                    flush=True,
                )
                result = {
                    "case_id": case_id,
                    "label": label,
                    "instruction": case["instruction"],
                    "user_text": case[field],
                    "source_dataset": case.get("source_dataset"),
                    "target_task": case.get("target_task"),
                    "attack": case.get("attack"),
                    **_run_chat(
                        host,
                        config,
                        args,
                        case["instruction"],
                        case[field],
                    ),
                }
                results.append(result)
                with predictions_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")

    output_path = run_dir / "ollama_benchmark.json"
    output_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "host": host,
                "input": args.input,
                "num_cases": len(cases),
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "thinking_enabled": args.think,
                "models": [config.model for config in models],
                "summary": _summarize_results(results),
                "predictions": predictions_path.name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.action == "list":
        _print_registry()
        return
    if args.action == "status":
        _print_status(args.host)
        return

    models = resolve_ollama_models(args.models)
    if args.action == "pull":
        _pull_models(models)
        return

    output_path = _run_benchmark(args.host, models, args)
    print(f"Saved benchmark: {output_path.resolve()}")


if __name__ == "__main__":
    main()
