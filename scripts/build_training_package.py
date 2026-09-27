from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "artifacts" / "nemotron_oracle_training_package"
DATASET_METADATA = {
    "title": "Nemotron Oracle Reasoning Traces",
    "id": "jahyee/nemotron-oracle-reasoning-traces",
    "subtitle": "Gold-conditioned symbolic traces for Nemotron reasoning SFT",
    "description": (
        "Oracle and deterministic reasoning traces generated from the NVIDIA "
        "Nemotron Model Reasoning Challenge training set, packaged for "
        "supervised fine-tuning experiments."
    ),
    "keywords": ["artificial intelligence", "nlp", "transformers"],
    "licenses": [{"name": "CC0-1.0"}],
}


TOKENIZE_SCRIPT = r'''from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


INPUT = Path("oracle_reasoning_traces.jsonl")
OUTPUT = Path("corpus_preprocessed.jsonl")
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
TOKEN_LIMIT = 8192


def main() -> None:
    chat_tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True,
    )

    written = 0
    with INPUT.open(encoding="utf-8") as input_file, OUTPUT.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            item = json.loads(line)
            messages = [{"role": "user", "content": item["prompt"] + PROMPT_SUFFIX}]
            prompt_ids = chat_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            completion_text = item["completion"].rstrip() + "\n</think>\n\\boxed{" + item["answer"] + "}<|im_end|>"
            completion_ids = chat_tokenizer.encode(completion_text, add_special_tokens=False)
            tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            if len(tokens) > TOKEN_LIMIT:
                tokens = tokens[:TOKEN_LIMIT]
                mask = mask[:TOKEN_LIMIT]
            if not any(mask):
                continue
            output_file.write(
                json.dumps(
                    {
                        "problem_id": item["id"],
                        "task_type": item["task_type"],
                        "method": item["method"],
                        "oracle": item.get("oracle", False),
                        "tokens": tokens,
                        "mask": mask,
                    }
                )
                + "\n"
            )
            written += 1
    print(f"Wrote {written} tokenized examples to {OUTPUT}")


if __name__ == "__main__":
    main()
'''

README = """# Nemotron Oracle Reasoning Traces

This package contains supervised reasoning traces for the NVIDIA Nemotron Model
Reasoning Challenge.

## Contents

- `oracle_reasoning_traces.jsonl`: 9,056 prompt/completion traces.
- `oracle_trace_report.json`: coverage summary.
- `solver_baseline.md`: no-gold deterministic solver report.
- `research_summary.md`: competition and experiment notes.
- `tokenize_oracle_traces.py`: Kaggle-side tokenizer script.

## Coverage

- No-gold deterministic traces: 8,364 / 9,500 = 88.04%.
- Gold-conditioned training traces: 9,056 / 9,500 = 95.33%.
- Symbolic equations improve from 581 / 1,555 to 1,273 / 1,555.

The cryptarithm oracle traces use the known training answer as a constraint to
recover latent symbol-to-digit mappings. They are intended for SFT, not as an
inference-time solver claim.

## Kaggle Use

1. Upload this folder as a Kaggle Dataset.
2. In a Kaggle notebook with the Nemotron model and required packages available,
   run:

```bash
python /kaggle/input/<dataset-slug>/tokenize_oracle_traces.py
```

3. Point the Huikang/Tinker training notebook's corpus path at the resulting
   `corpus_preprocessed.jsonl`.
4. Train rank-32 LoRA with the existing 8192-token setup.
"""


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    copy_required(
        ROOT / "data" / "generated" / "oracle_reasoning_traces.jsonl",
        PACKAGE_DIR / "oracle_reasoning_traces.jsonl",
    )
    copy_required(
        ROOT / "reports" / "oracle_trace_report.json",
        PACKAGE_DIR / "oracle_trace_report.json",
    )
    copy_required(
        ROOT / "reports" / "solver_baseline.md",
        PACKAGE_DIR / "solver_baseline.md",
    )
    copy_required(
        ROOT / "reports" / "research_summary.md",
        PACKAGE_DIR / "research_summary.md",
    )

    (PACKAGE_DIR / "tokenize_oracle_traces.py").write_text(TOKENIZE_SCRIPT, encoding="utf-8")
    (PACKAGE_DIR / "README.md").write_text(README, encoding="utf-8")
    (PACKAGE_DIR / "dataset-metadata.json").write_text(
        json.dumps(DATASET_METADATA, indent=2), encoding="utf-8"
    )

    print(f"Built training package at {PACKAGE_DIR}")
    for path in sorted(PACKAGE_DIR.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
