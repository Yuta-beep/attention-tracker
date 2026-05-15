import argparse
import json
from pathlib import Path

from .attention import extract_last_token_instruction_attention
from .config import get_model_config
from .modeling import load_model_bundle
from .prompting import build_chat_prompt
from .token_span import find_token_indices_for_substring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2_1.5b")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_model_config(args.model)
    bundle = load_model_bundle(config)

    prompt = build_chat_prompt(
        tokenizer=bundle.tokenizer,
        instruction=args.instruction,
        user_text=args.text,
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
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
