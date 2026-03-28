from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import TARGET_COLUMN, load_training_data, take_smoke_subset
from features import build_features
from metrics import evaluate_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hallucination detector")
    parser.add_argument("--input", required=True, help="Path to CSV/JSONL with labels")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model/metrics")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Feature extraction device")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-check", action="store_true", help="Train on a few rows only")
    parser.add_argument("--smoke-rows", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        except Exception as e:
            raise RuntimeError(
                "CUDA mode requires torch with CUDA. Use --device cpu for CUDA-free run."
            ) from e

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(args.input)
    if args.smoke_check:
        df = take_smoke_subset(df, args.smoke_rows)

    X = build_features(df, device=args.device)
    y = df[TARGET_COLUMN].to_numpy()

    stratify = y if len(np.unique(y)) > 1 and len(y) >= 10 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_score = model.predict_proba(X_valid)[:, 1]
    infer_s = time.perf_counter() - t1

    metrics = evaluate_binary(y_valid, y_score)
    metrics.update(
        {
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid)),
            "train_seconds": train_s,
            "valid_infer_seconds": infer_s,
            "device": args.device,
            "smoke_check": bool(args.smoke_check),
        }
    )

    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(X.columns), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
