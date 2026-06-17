import os
import sys

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


REPOSITORIES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]


def main() -> int:
    load_dotenv(".env.local")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set in .env.local.", file=sys.stderr)
        return 2

    failures = 0
    for repository in REPOSITORIES:
        try:
            path = hf_hub_download(repository, "config.json", token=token)
            print(f"OK     {repository}")
            print(f"       {path}")
        except Exception as exc:
            failures += 1
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"FAILED {repository}")
            print(f"       {type(exc).__name__}: {first_line}")

    print(f"\nResult: {len(REPOSITORIES) - failures}/{len(REPOSITORIES)} accessible")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
