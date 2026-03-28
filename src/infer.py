from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from data import load_table
from features import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hallucination inference")
    parser.add_argument("--input", required=True, help="CSV or JSONL with prompt/model_answer")
    parser.add_argument("--model", default="artifacts/model.pkl", help="Path to trained model.pkl")
    parser.add_argument("--output", required=True, help="Output CSV or JSONL path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Feature extraction device")
    return parser.parse_args()


def _write_output(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".jsonl", ".json"}:
        with path.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    args = parse_args()
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = load_table(args.input)
    X = build_features(df, device=args.device)
    score = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["hallucination_score"] = score
    out["is_hallucination_pred"] = (score >= 0.5).astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(out, out_path)

    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
