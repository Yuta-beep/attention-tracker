# attention-tracker

## Recommended Ollama inference models

The repository manages three recent inference models separately from the
Transformers-based attention extraction experiments:

| Registry key | Ollama model | Default context | Role |
|---|---|---:|---|
| `qwen3.5_9b` | `qwen3.5:9b` | 32768 | General reasoning, Japanese, coding, vision, and tools |
| `gemma4_12b` | `gemma4:12b` | 32768 | Independent multimodal reasoning comparison |
| `gpt_oss_20b` | `gpt-oss:20b` | 16384 | Reasoning, structured output, coding, and tools |

These models are not added to `MODEL_REGISTRY`. Qwen 3.5 mixes linear and
full attention, Gemma 4 requires a newer multimodal model implementation, and
gpt-oss uses MXFP4 MoE weights. Ollama also does not expose the per-head
attention tensors required by Attention Tracker.

After pulling this repository on the GPU server:

```bash
cd ~/Developer/attention-tracker

~/.local/bin/uv sync
~/.local/bin/uv run manage-ollama-models list
~/.local/bin/uv run manage-ollama-models status
~/.local/bin/uv run manage-ollama-models pull --models all
```

Run the same short non-thinking benchmark across all three models:

```bash
~/.local/bin/uv run manage-ollama-models benchmark \
  --models all \
  --num-predict 128 \
  --experiment-title recommended-models-non-thinking
```

Run a reasoning benchmark:

```bash
~/.local/bin/uv run manage-ollama-models benchmark \
  --models all \
  --think \
  --num-predict 512 \
  --prompt "Solve the problem carefully and verify the result: ..." \
  --experiment-title recommended-models-thinking
```

Benchmark output is written under:

```text
outputs/<timestamp>_<experiment-title>/ollama_benchmark.json
```

## Prompt-injection detector baselines

The paper baselines are implemented separately from Attention Tracker under
`last_token_attention/baselines/`. Every detector consumes the same JSONL
cases used by Attention Tracker and writes one continuous `attack_score` for
the normal and attacked version of every case.

Available detectors:

```text
protect_ai    Protect AI DeBERTa classifier
prompt_guard  Meta Prompt Guard classifier
llm_based     target LLM judges whether the input is safe
known_answer  target LLM is tested with the HELLO control instruction
```

Run the trained classifiers:

```bash
uv run evaluate-protect-ai \
  --input data.eval.open_prompt_injection.jsonl

uv run evaluate-prompt-guard \
  --input data.eval.open_prompt_injection.jsonl
```

Prompt Guard is gated and requires an accepted Hugging Face license and an
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

Run target-LLM-dependent baselines:

```bash
uv run evaluate-llm-based \
  --model qwen2_1.5b \
  --input data.eval.open_prompt_injection.jsonl

uv run evaluate-known-answer \
  --model qwen2_1.5b \
  --input data.eval.open_prompt_injection.jsonl
```

`evaluate-baseline-detector --detector ...` remains available as a common
orchestration entry point.

Run all paper baselines across the seven Attention Tracker target models:

```bash
uv run evaluate-all-baselines \
  --input data.eval.open_prompt_injection.jsonl \
  --models all \
  --detectors all \
  --experiment-title open-prompt-injection-all-baselines
```

This creates 16 jobs: Protect AI and Prompt Guard once each, plus LLM-based
and Known-answer for each of the seven target LLMs. Use `--dry-run` to inspect
the complete execution manifest without loading models.

Each run writes:

```text
outputs/<timestamp>_<title>/
  predictions.jsonl
  metrics.json
  README.txt
```
