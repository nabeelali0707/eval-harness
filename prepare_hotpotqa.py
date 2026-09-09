"""Prepare corpus + eval questions from a raw HotpotQA JSON file.

The input JSON is expected to match the official HotpotQA format:
each record has 'question', 'answer', 'type', 'support_facts', and 'context'.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def normalize_doc_id(title: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return safe or "doc"


def build_corpus(records: list[dict]) -> tuple[list[dict], dict[str, str]]:
    title_to_id: dict[str, str] = {}
    corpus: list[dict] = []
    seen_titles: set[str] = set()

    for record in records:
        for title, sentences in record["context"]:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            doc_id = normalize_doc_id(title)
            base_id = doc_id
            counter = 1
            while doc_id in title_to_id.values():
                doc_id = f"{base_id}_{counter}"
                counter += 1
            title_to_id[title] = doc_id
            corpus.append(
                {
                    "doc_id": doc_id,
                    "text": " ".join(sentences),
                    "source": title,
                    "metadata": {"original_title": title},
                }
            )

    return corpus, title_to_id


def build_questions(records: list[dict], title_to_id: dict[str, str]) -> list[dict]:
    questions = []
    for idx, record in enumerate(records):
        gold_doc_ids = []
        for title, _ in record["supporting_facts"]:
            doc_id = title_to_id.get(title)
            if doc_id and doc_id not in gold_doc_ids:
                gold_doc_ids.append(doc_id)

        questions.append(
            {
                "id": f"hotpotqa_{idx:04d}",
                "question": record["question"],
                "expected_answer": record["answer"],
                "gold_doc_ids": gold_doc_ids,
                "type": record.get("type", "unknown"),
                "level": record.get("level", "unknown"),
            }
        )
    return questions


def main(input_path: Path, sample_size: int | None, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {input_path} ...")
    with input_path.open(encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records")

    if sample_size and sample_size < len(records):
        random.seed(seed)
        records = random.sample(records, sample_size)

    corpus, title_to_id = build_corpus(records)
    questions = build_questions(records, title_to_id)

    corpus_path = output_dir / "corpus.json"
    questions_path = output_dir / "eval_questions.json"

    corpus_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    questions_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    print(f"Wrote {len(corpus)} corpus chunks to {corpus_path}")
    print(f"Wrote {len(questions)} eval questions to {questions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json")
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--output", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(Path(args.input), args.sample, Path(args.output), args.seed)
