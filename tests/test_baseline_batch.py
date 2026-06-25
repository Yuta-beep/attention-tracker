from argparse import Namespace
from pathlib import Path

from last_token_attention.baselines.batch_cli import (
    build_jobs,
    command_for_job,
    split_detectors,
)


MODELS = [
    "qwen2_1.5b",
    "qwen2_7b",
    "llama3_8b",
    "mistral_7b",
    "phi3_mini",
    "gemma2_2b",
    "gemma2_9b",
]


def test_all_detectors_create_sixteen_jobs_for_seven_models():
    jobs = build_jobs(split_detectors("all"), MODELS)

    assert len(jobs) == 16
    assert sum(job.detector == "protect_ai" for job in jobs) == 1
    assert sum(job.detector == "prompt_guard" for job in jobs) == 1
    assert sum(job.detector == "llm_based" for job in jobs) == 7
    assert sum(job.detector == "known_answer" for job in jobs) == 7


def test_model_dependent_command_contains_target_model():
    args = Namespace(
        input="data.jsonl",
        separator="\n\n",
        torch_dtype="",
        load_in_4bit=False,
        no_chat_template=False,
        max_length=512,
    )
    job = next(
        job
        for job in build_jobs(["llm_based"], ["gemma2_9b"])
        if job.model_key == "gemma2_9b"
    )

    command = command_for_job(args, Path("outputs/run"), job)

    assert "--model" in command
    assert command[command.index("--model") + 1] == "gemma2_9b"
