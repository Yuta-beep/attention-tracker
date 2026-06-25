from pathlib import Path


MODULES = (
    "protect_ai.py",
    "prompt_guard.py",
    "llm_based.py",
    "known_answer.py",
)


def test_detector_modules_have_python_module_entrypoints():
    baseline_dir = (
        Path(__file__).parents[1]
        / "src"
        / "last_token_attention"
        / "baselines"
    )

    for filename in MODULES:
        source = (baseline_dir / filename).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source
        assert "\n    main()\n" in source
