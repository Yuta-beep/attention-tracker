from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DetectionExample:
    case_id: str
    instruction: str
    text: str
    true_label: str
    metadata: dict


class Detector(Protocol):
    name: str
    model_id: str

    def score(self, example: DetectionExample) -> float:
        """Return a continuous score where larger means more likely attack."""


def normalize_case(raw_case: dict, default_separator: str = "\n\n") -> tuple[DetectionExample, DetectionExample]:
    instruction = str(raw_case["instruction"])
    separator = str(raw_case.get("separator", default_separator))
    normal_text = raw_case.get("normal_text", raw_case.get("base_text"))
    if normal_text is None:
        raise ValueError("Each case must contain normal_text or base_text.")

    injection_text = raw_case.get("injection_text")
    if injection_text is None:
        raise ValueError("Each case must contain injection_text.")
    injected_text = raw_case.get("injected_text")
    if injected_text is None:
        injected_text = f"{normal_text}{separator}{injection_text}"

    case_id = str(raw_case.get("id", "unknown-case"))
    metadata = {
        key: value
        for key, value in raw_case.items()
        if key
        not in {
            "id",
            "instruction",
            "base_text",
            "normal_text",
            "injected_text",
            "injection_text",
            "separator",
        }
    }
    return (
        DetectionExample(
            case_id=case_id,
            instruction=instruction,
            text=str(normal_text),
            true_label="normal",
            metadata=metadata,
        ),
        DetectionExample(
            case_id=case_id,
            instruction=instruction,
            text=str(injected_text),
            true_label="attack",
            metadata=metadata,
        ),
    )


def load_detection_examples(path: str | Path, default_separator: str = "\n\n") -> list[DetectionExample]:
    examples = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw_case = json.loads(line)
            examples.extend(normalize_case(raw_case, default_separator))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL case at line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError("Input JSONL did not contain any cases.")
    return examples
