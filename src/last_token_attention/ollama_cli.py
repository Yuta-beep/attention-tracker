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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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


def _benchmark_model(
    host: str,
    config: OllamaModelConfig,
    args: argparse.Namespace,
) -> dict:
    context = args.num_ctx or config.default_context
    payload = {
        "model": config.model,
        "prompt": args.prompt,
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
    response = _request_json(host, "/api/generate", payload=payload)
    wall_seconds = time.perf_counter() - started
    return {
        "model_key": config.key,
        "model": config.model,
        "role": config.role,
        "context": context,
        "thinking_enabled": args.think,
        "wall_seconds": wall_seconds,
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
        "response": response.get("response", ""),
        "thinking": response.get("thinking", ""),
    }


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

    run_dir = create_run_dir(args.output_root, args.experiment_title)
    results = []
    for index, config in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] Benchmarking {config.model}", flush=True)
        result = _benchmark_model(host, config, args)
        results.append(result)
        speed = result["generation_tokens_per_second"]
        speed_text = f"{speed:.2f} tok/s" if speed is not None else "N/A"
        print(f"[{index}/{len(models)}] {config.model}: {speed_text}", flush=True)

    output_path = run_dir / "ollama_benchmark.json"
    output_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "host": host,
                "prompt": args.prompt,
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "thinking_enabled": args.think,
                "results": results,
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
