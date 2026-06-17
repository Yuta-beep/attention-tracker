import argparse
import json
from pathlib import Path

from .attention import extract_last_token_attention_to_spans, sum_attention_scores
from .config import resolve_model_config
from .modeling import load_model_bundle
from .paper_analysis import generate_paper_analysis
from .plots import (
    plot_case_heatmaps,
    plot_case_token_position_heatmaps,
    plot_case_totals,
    plot_mean_head_distraction,
    plot_summary_totals,
    plot_top_distracted_heads,
)
from .prompting import build_chat_prompt, build_injected_user_text
from .runs import create_run_dir
from .token_span import find_token_indices_for_substring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen2_1.5b",
        help="Registry key or Hugging Face model id. Use --model-id to pass an explicit id.",
    )
    parser.add_argument("--model-id", default="", help="Explicit Hugging Face model id; overrides --model.")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument("--input", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--base-text", default="")
    parser.add_argument("--injection-text", default="")
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output", default="")
    parser.add_argument("--plot-dir", default="")
    parser.add_argument(
        "--detailed-plots",
        action="store_true",
        help="Also generate three plots for every individual case.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="compare-last-token-attention")
    return parser.parse_args()


def _load_cases(args: argparse.Namespace) -> list[dict]:
    if args.input:
        cases = []
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        if not cases:
            raise ValueError("Input JSONL did not contain any cases.")
        return cases

    if not args.instruction or not args.base_text or not args.injection_text:
        raise ValueError(
            "Provide either --input or all of --instruction, --base-text, and --injection-text."
        )

    return [
        {
            "id": "case-0001",
            "instruction": args.instruction,
            "base_text": args.base_text,
            "injection_text": args.injection_text,
            "separator": args.separator,
        }
    ]


def _normalize_case(raw_case: dict, default_separator: str) -> dict:
    instruction = raw_case["instruction"]
    separator = raw_case.get("separator", default_separator)
    if "base_text" in raw_case and "injection_text" in raw_case:
        base_text = raw_case["base_text"]
        injection_text = raw_case["injection_text"]
        normal_text = raw_case.get("normal_text", base_text)
        injected_text = raw_case.get(
            "injected_text",
            build_injected_user_text(base_text, injection_text, separator=separator),
        )
    else:
        normal_text = raw_case["normal_text"]
        injected_text = raw_case["injected_text"]
        injection_text = raw_case["injection_text"]

    return {
        "id": raw_case.get("id", "unknown-case"),
        "instruction": instruction,
        "normal_text": normal_text,
        "injected_text": injected_text,
        "injection_text": injection_text,
        "separator": separator,
    }


def _analyze_case(bundle, config, case: dict) -> dict:
    normal_prompt = build_chat_prompt(
        tokenizer=bundle.tokenizer,
        instruction=case["instruction"],
        user_text=case["normal_text"],
        uses_chat_template=config.uses_chat_template,
        system_role_supported=config.system_role_supported,
    )
    injected_prompt = build_chat_prompt(
        tokenizer=bundle.tokenizer,
        instruction=case["instruction"],
        user_text=case["injected_text"],
        uses_chat_template=config.uses_chat_template,
        system_role_supported=config.system_role_supported,
    )

    instruction_span_normal = find_token_indices_for_substring(
        bundle.tokenizer, normal_prompt, case["instruction"]
    )
    instruction_span_injected = find_token_indices_for_substring(
        bundle.tokenizer, injected_prompt, case["instruction"]
    )
    injection_span = find_token_indices_for_substring(
        bundle.tokenizer, injected_prompt, case["injection_text"]
    )

    normal_result = extract_last_token_attention_to_spans(
        bundle,
        normal_prompt,
        {"instruction": instruction_span_normal},
    )
    injected_result = extract_last_token_attention_to_spans(
        bundle,
        injected_prompt,
        {
            "instruction": instruction_span_injected,
            "injection": injection_span,
        },
    )

    normal_instruction = normal_result.per_layer_head_scores["instruction"]
    attack_instruction = injected_result.per_layer_head_scores["instruction"]
    attack_injection = injected_result.per_layer_head_scores["injection"]

    return {
        "id": case["id"],
        "instruction": case["instruction"],
        "normal_text": case["normal_text"],
        "injected_text": case["injected_text"],
        "injection_text": case["injection_text"],
        "normal_prompt": normal_prompt,
        "injected_prompt": injected_prompt,
        "normal_token_texts": normal_result.token_texts,
        "injected_token_texts": injected_result.token_texts,
        "normal_token_scores": normal_result.per_layer_token_scores.tolist(),
        "injected_token_scores": injected_result.per_layer_token_scores.tolist(),
        "instruction_token_indices_normal": normal_result.span_token_indices["instruction"],
        "instruction_token_indices_injected": injected_result.span_token_indices["instruction"],
        "injection_token_indices": injected_result.span_token_indices["injection"],
        "normal_instruction_total": sum_attention_scores(normal_instruction),
        "attack_instruction_total": sum_attention_scores(attack_instruction),
        "attack_injection_total": sum_attention_scores(attack_injection),
        "instruction_total_delta": sum_attention_scores(attack_instruction) - sum_attention_scores(normal_instruction),
        "normal_instruction_scores": normal_instruction.tolist(),
        "attack_instruction_scores": attack_instruction.tolist(),
        "attack_injection_scores": attack_injection.tolist(),
    }


def _build_summary(results: list[dict]) -> dict:
    normal_totals = [row["normal_instruction_total"] for row in results]
    attack_instruction_totals = [row["attack_instruction_total"] for row in results]
    attack_injection_totals = [row["attack_injection_total"] for row in results]
    deltas = [row["instruction_total_delta"] for row in results]

    return {
        "num_cases": len(results),
        "mean_normal_instruction_total": sum(normal_totals) / len(normal_totals),
        "mean_attack_instruction_total": sum(attack_instruction_totals) / len(attack_instruction_totals),
        "mean_attack_injection_total": sum(attack_injection_totals) / len(attack_injection_totals),
        "mean_instruction_total_delta": sum(deltas) / len(deltas),
        "num_cases_attack_instruction_lower_than_normal": sum(
            attack < normal for normal, attack in zip(normal_totals, attack_instruction_totals)
        ),
        "num_cases_attack_injection_higher_than_attack_instruction": sum(
            inj > attack for inj, attack in zip(attack_injection_totals, attack_instruction_totals)
        ),
    }


def _generate_plots(
    results: list[dict], plot_dir: Path, detailed_plots: bool = False
) -> None:
    generate_paper_analysis(results, plot_dir)

    if detailed_plots:
        detailed_dir = plot_dir / "details"
        for case in results:
            case_dir = detailed_dir / case["id"]
            plot_case_totals(case, case_dir / "attention_totals.png")
            plot_case_heatmaps(case, case_dir / "head_attention_heatmaps.png")
            plot_case_token_position_heatmaps(
                case, case_dir / "token_attention_by_position.png"
            )


def _resolve_run_path(run_dir: Path, raw_path: str, default_name: str) -> Path:
    if not raw_path:
        return run_dir / default_name
    return run_dir / Path(raw_path).name


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = create_run_dir(args.output_root, args.experiment_title)
    output_path = _resolve_run_path(run_dir, args.output, "results.json")
    plot_dir = _resolve_run_path(run_dir, args.plot_dir, "plots")
    return output_path, plot_dir, run_dir


def main() -> None:
    args = parse_args()
    output_path, plot_dir, run_dir = _resolve_output_paths(args)

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

    payload = {
        "experiment_title": args.experiment_title,
        "run_dir": str(run_dir) if run_dir else "",
        "model_key": config.key,
        "model_id": config.model_id,
        "summary": _build_summary(results),
        "cases": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if plot_dir is not None:
        _generate_plots(results, plot_dir, args.detailed_plots)

    if run_dir is not None:
        (run_dir / "README.txt").write_text(
            "Generated files:\n"
            "- results.json\n"
            "- plots/01_figure2a_head_distraction.png\n"
            "- plots/02_figure2b_token_shift.png\n"
            "- plots/03_figure3_attack_distributions.png\n"
            "- plots/04_figure8_candidate_scores_k4.png\n"
            "- plots/k_sweep/*.png\n"
            "- plots/k_sweep/k_sweep_metrics.json\n"
            "- plots/paper_metrics.json\n",
            encoding="utf-8",
        )

    print(f"Saved results: {output_path.resolve()}")
    if plot_dir is not None:
        print(f"Saved plots: {plot_dir.resolve()}")
    if run_dir is not None:
        print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
