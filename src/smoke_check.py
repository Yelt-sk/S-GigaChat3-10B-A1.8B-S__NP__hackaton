from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU smoke-check for train+infer pipeline")
    p.add_argument("--input", default="knowledge_bench_public.csv")
    p.add_argument("--workdir", default="smoke_artifacts")
    p.add_argument("--rows", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "src/train.py",
        "--input",
        args.input,
        "--output-dir",
        str(workdir),
        "--device",
        "cpu",
        "--smoke-check",
        "--smoke-rows",
        str(args.rows),
    ]
    infer_cmd = [
        sys.executable,
        "src/infer.py",
        "--input",
        args.input,
        "--model",
        str(workdir / "model.pkl"),
        "--output",
        str(workdir / "predictions.csv"),
        "--device",
        "cpu",
    ]

    print("Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)
    print("Running:", " ".join(infer_cmd))
    subprocess.run(infer_cmd, check=True)
    print(f"Smoke-check is OK. Artifacts: {workdir}")


if __name__ == "__main__":
    main()
