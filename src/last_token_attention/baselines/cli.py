import argparse
import json
from pathlib import Path
from time import perf_counter

from ..config import resolve_model_config
from ..runs import create_run_dir
from .common import load_detection_examples
from .metrics import summarize


DETECTOR_NAMES = ("protect_ai", "prompt_guard", "llm_based", "known_answer")


def parse_args(fixed_detector: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a prompt-injection detector baseline on shared JSONL cases."
    )
    if fixed_detector is None:
        parser.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    else:
        parser.set_defaults(detector=fixed_detector)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--model-id",
        default="",
        help="Model override: classifier model for trained detectors, target LLM otherwise.",
    )
    if fixed_detector is None or fixed_detector in {"llm_based", "known_answer"}:
        parser.add_argument(
            "--model",
            default="qwen2_1.5b",
            help="Target LLM registry key for llm_based and known_answer.",
        )
        parser.add_argument(
            "--torch-dtype",
            choices=["float16", "bfloat16", "float32"],
            default="",
        )
        parser.add_argument("--load-in-4bit", action="store_true")
        parser.add_argument("--no-chat-template", action="store_true")
    if fixed_detector is None or fixed_detector in {"protect_ai", "prompt_guard"}:
        parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="")
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    if args.detector == "protect_ai":
        from .protect_ai import DEFAULT_MODEL_ID, ProtectAIDetector

        return ProtectAIDetector(
            model_id=args.model_id or DEFAULT_MODEL_ID,
            max_length=getattr(args, "max_length", 512),
        )
    if args.detector == "prompt_guard":
        from .prompt_guard import DEFAULT_MODEL_ID, PromptGuardDetector

        return PromptGuardDetector(
            model_id=args.model_id or DEFAULT_MODEL_ID,
            max_length=getattr(args, "max_length", 512),
        )

    config = resolve_model_config(
        args.model_id or getattr(args, "model", "qwen2_1.5b"),
        torch_dtype=getattr(args, "torch_dtype", "") or None,
        load_in_4bit=True if getattr(args, "load_in_4bit", False) else None,
        uses_chat_template=(
            False if getattr(args, "no_chat_template", False) else None
        ),
    )
    if args.detector == "llm_based":
        from .llm_based import LLMBasedDetector

        return LLMBasedDetector(config)
    from .known_answer import KnownAnswerDetector

    return KnownAnswerDetector(config)


def evaluate(detector, examples) -> list[dict]:
    rows = []
    for index, example in enumerate(examples, start=1):
        started = perf_counter()
        attack_score = detector.score(example)
        latency_ms = (perf_counter() - started) * 1000
        rows.append(
            {
                "prediction_index": index,
                "case_id": example.case_id,
                "detector": detector.name,
                "detector_model_id": detector.model_id,
                "true_label": example.true_label,
                "attack_score": float(attack_score),
                "latency_ms": latency_ms,
                "metadata": example.metadata,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    title = args.experiment_title or f"baseline-{args.detector}"
    run_dir = create_run_dir(args.output_root, title)
    examples = load_detection_examples(args.input, args.separator)
    detector = build_detector(args)
    rows = evaluate(detector, examples)

    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = {
        "detector": detector.name,
        "detector_model_id": detector.model_id,
        "input": str(Path(args.input).resolve()),
        "summary": summarize(rows),
        "predictions": str(predictions_path),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- predictions.jsonl: one continuous attack score per normal/attack input\n"
        "- metrics.json: micro AUROC, optional grouped macro AUROC, and latency\n",
        encoding="utf-8",
    )
    print(f"Predictions: {predictions_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Run directory: {run_dir.resolve()}")


def main_for_detector(detector_name: str) -> None:
    if detector_name not in DETECTOR_NAMES:
        raise ValueError(f"Unknown detector: {detector_name}")
    run(parse_args(fixed_detector=detector_name))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
