from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"prompt", "model_answer"}
TARGET_COLUMN = "is_hallucination"


class DataValidationError(ValueError):
    """Raised when input data does not follow expected schema."""


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )


def _to_binary_labels(series: pd.Series) -> pd.Series:
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        1: 1,
        0: 0,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        bad_values = series[converted.isna()].unique().tolist()
        raise DataValidationError(
            f"Unsupported values in '{TARGET_COLUMN}': {bad_values}"
        )
    return converted.astype(int)


def load_table(path: str | Path) -> pd.DataFrame:
    """Load CSV or JSONL file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise DataValidationError(f"Unsupported file extension: {suffix}")

    _validate_columns(df, REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    return df


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load labeled training data and normalize target to {0,1}."""
    df = load_table(path)
    _validate_columns(df, [TARGET_COLUMN])
    df[TARGET_COLUMN] = _to_binary_labels(df[TARGET_COLUMN])
    return df


def take_smoke_subset(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Small deterministic subset for quick local checks."""
    n = max(1, int(n))
    return df.head(n).reset_index(drop=True)
