"""Run the complete five-mode ablation sequentially with resumable checkpoints."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.keep_awake import inhibit_sleep
from src.ollama_client import OllamaError

MODE_ORDER = [
    "dense_only",
    "bm25_only",
    "hybrid_rrf",
    "hybrid_rerank",
    "hybrid_rewrite_rerank",
]


def mode_csv_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f"{mode}.csv"


def pending_modes(output_dir: Path, modes: list[str]) -> list[str]:
    """Return modes whose CSV has not been written yet."""
    return [mode for mode in modes if not mode_csv_path(output_dir, mode).exists()]


def run_ablation(
    modes: list[str],
    output_dir: Path,
    *,
    no_generator: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[str]:
    """Run each mode sequentially, resuming existing checkpoints.

    Returns the modes that produced a completed CSV. A mode that fails with
    an unrecoverable OllamaError aborts the remaining sequence so the caller
    can inspect and resume later.
    """
    from run_eval import main as run_eval_main

    completed: list[str] = []
    for mode in modes:
        csv_path = mode_csv_path(output_dir, mode)
        if csv_path.exists():
            print(f"[ablation] {mode}: CSV already exists, skipping.")
            completed.append(mode)
            continue

        config_path = Path("configs") / f"{mode}.yaml"
        print(f"[ablation] {mode}: starting ({config_path})")
        try:
            run_eval_main(
                config_path,
                Path("data/eval_questions.json"),
                output_dir,
                None,
                no_generator,
                True,
                False,
                max_retries,
                retry_delay,
            )
        except OllamaError as error:
            print(f"[ablation] {mode}: failed after retries: {error}")
            print("[ablation] Stopping; fix Ollama and rerun to resume.")
            return completed

        completed.append(mode)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/final_local_ollama_150q",
        help="Output directory shared by all five modes",
    )
    parser.add_argument(
        "--no-generator",
        action="store_true",
        help="Run retrieval-only versions of every mode",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    modes = pending_modes(output_dir, MODE_ORDER)
    if not modes:
        print("[ablation] All five mode CSVs already exist. Nothing to do.")
        return

    print(f"[ablation] Modes to run: {', '.join(modes)}")
    started = time.perf_counter()
    with inhibit_sleep():
        completed = run_ablation(
            modes,
            output_dir,
            no_generator=args.no_generator,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
    elapsed_min = (time.perf_counter() - started) / 60
    print(
        f"[ablation] Finished {len(completed)}/{len(MODE_ORDER)} modes "
        f"in {elapsed_min:.1f} minutes."
    )
    if len(completed) < len(MODE_ORDER):
        print(
            "[ablation] Incomplete. Rerun the same command to resume "
            "from existing checkpoints."
        )


if __name__ == "__main__":
    main()
