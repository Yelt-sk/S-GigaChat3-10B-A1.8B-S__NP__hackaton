"""Training entrypoint for hallucination detector baseline."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hallucination detector")
    parser.add_argument(
        "--data-path",
        default="knowledge_bench_public.csv",
        help="Path to CSV dataset (default: knowledge_bench_public.csv)",
    )
    return parser.parse_args()


def validate_data_path(data_path: str) -> Path:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.resolve()}\n"
            "Hint: pass --data-path <path/to/knowledge_bench_public.csv> or place "
            "knowledge_bench_public.csv in the project root."
        )
    return path


def main() -> None:
    args = parse_args()
    csv_path = validate_data_path(args.data_path)
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")


if __name__ == "__main__":
    main()
