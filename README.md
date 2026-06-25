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
