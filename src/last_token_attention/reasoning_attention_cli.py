import argparse
import json
from pathlib import Path

import torch

from .attention import extract_generated_reasoning_attention_to_spans
from .compare_cli import _load_cases, _normalize_case
from .config import resolve_model_config
from .modeling import load_model_bundle
from .prompting import build_chat_prompt
from .runs import create_run_dir, slugify
from .token_span import find_token_indices_for_substring


REASONING_MODELS = ("qwen3.5_9b", "gemma4_12b", "gpt_oss_20b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reasoning and save per-generated-token attention to the "
            "original instruction."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", choices=REASONING_MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="reasoning-attention")
    return parser.parse_args()


def _run_variant(
    bundle,
    config,
    case: dict,
    label: str,
    text: str,
    output_dir: Path,
    max_new_tokens: int,
) -> dict:
    prompt = build_chat_prompt(
        tokenizer=bundle.tokenizer,
        instruction=case["instruction"],
        user_text=text,
        uses_chat_template=config.uses_chat_template,
        system_role_supported=config.system_role_supported,
        reasoning_enabled=True,
    )
    instruction_span = find_token_indices_for_substring(
        bundle.tokenizer,
        prompt,
        case["instruction"],
    )
    result = extract_generated_reasoning_attention_to_spans(
        bundle,
        prompt,
        {"instruction": instruction_span},
        max_new_tokens=max_new_tokens,
    )
    tensor_path = output_dir / f"{slugify(case['id'])}_{label}.pt"
    torch.save(
        {
            "case_id": case["id"],
            "label": label,
            "model_key": config.key,
            "model_id": config.model_id,
            "prompt": prompt,
            "generated_text": result.generated_text,
            "generated_token_ids": result.generated_token_ids,
            "generated_token_texts": result.generated_token_texts,
            "active_layer_indices": result.active_layer_indices,
            "instruction_token_indices": instruction_span.token_indices,
            "instruction_attention": result.per_token_layer_head_scores["instruction"],
        },
        tensor_path,
    )
    scores = result.per_token_layer_head_scores["instruction"]
    per_token_focus = torch.nanmean(scores, dim=(1, 2))
    return {
        "case_id": case["id"],
        "label": label,
        "model_key": config.key,
        "model_id": config.model_id,
        "num_generated_tokens": len(result.generated_token_ids),
        "generated_text": result.generated_text,
        "active_attention_layers": result.active_layer_indices,
        "mean_instruction_focus": float(torch.nanmean(scores).item()),
        "per_token_instruction_focus": per_token_focus.tolist(),
        "tensor_path": str(tensor_path),
    }


def main() -> None:
    args = parse_args()
    config = resolve_model_config(args.model)
    bundle = load_model_bundle(config)
    raw_cases = _load_cases(args)
    cases = [_normalize_case(case, args.separator) for case in raw_cases]
    if args.limit:
        cases = cases[: args.limit]

    run_dir = create_run_dir(args.output_root, args.experiment_title)
    tensor_dir = run_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        rows.append(
            _run_variant(
                bundle,
                config,
                case,
                "normal",
                case["normal_text"],
                tensor_dir,
                args.max_new_tokens,
            )
        )
        rows.append(
            _run_variant(
                bundle,
                config,
                case,
                "attack",
                case["injected_text"],
                tensor_dir,
                args.max_new_tokens,
            )
        )

    summary_path = run_dir / "reasoning_attention.jsonl"
    summary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "model_key": config.key,
        "model_id": config.model_id,
        "input": str(Path(args.input).resolve()),
        "num_cases": len(cases),
        "num_variants": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "summary_path": str(summary_path),
        "tensor_dir": str(tensor_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path.resolve()}")
    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
