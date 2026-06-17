import argparse
import json
from pathlib import Path

from .attention import extract_last_token_instruction_attention
from .config import resolve_model_config
from .modeling import load_model_bundle
from .prompting import build_chat_prompt
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
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_model_config(
        args.model_id or args.model,
        torch_dtype=args.torch_dtype or None,
        load_in_4bit=True if args.load_in_4bit else None,
        uses_chat_template=False if args.no_chat_template else None,
        system_role_supported=False if args.no_system_role else None,
    )
    bundle = load_model_bundle(config)

    prompt = build_chat_prompt(
        tokenizer=bundle.tokenizer,
        instruction=args.instruction,
        user_text=args.text,
        uses_chat_template=config.uses_chat_template,
        system_role_supported=config.system_role_supported,
    )
    span = find_token_indices_for_substring(bundle.tokenizer, prompt, args.instruction)
    result = extract_last_token_instruction_attention(bundle, prompt, span)

    payload = {
        "model_key": config.key,
        "model_id": config.model_id,
        "prompt": result.prompt,
        "instruction_token_indices": result.instruction_token_indices,
        "num_layers": int(result.per_layer_head_scores.shape[0]),
        "num_heads": int(result.per_layer_head_scores.shape[1]),
        "per_layer_head_scores": result.per_layer_head_scores.tolist(),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved output: {output_path.resolve()}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
