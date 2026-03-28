from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def preprocess(text: str) -> str:
    """Lightweight normalization used by feature extraction."""
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


@dataclass
class FeatureAccumulator:
    """Collects scalar features and returns them as a flat dictionary."""

    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        if isinstance(value, (bool, np.bool_)):
            self.values[name] = float(value)
            return
        if value is None:
            self.values[name] = 0.0
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                self.values[name] = 0.0
            else:
                self.values[name] = float(value)
            return
        self.values[name] = 0.0

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


def _uncertainty_features(prompt: str, answer: str) -> dict[str, float]:
    p_words = set(WORD_RE.findall(prompt))
    a_words = WORD_RE.findall(answer)
    if not a_words:
        return {
            "unc_len_tokens": 0.0,
            "unc_rare_token_ratio": 1.0,
            "unc_long_token_ratio": 0.0,
            "unc_prompt_overlap_ratio": 0.0,
            "unc_symbol_ratio": 0.0,
            "unc_uppercase_ratio": 0.0,
        }

    token_count = len(a_words)
    overlap = sum(1 for t in a_words if t in p_words)
    long_tokens = sum(1 for t in a_words if len(t) >= 10)
    rare_like = sum(1 for t in a_words if len(t) <= 2 or any(ch.isdigit() for ch in t))

    symbol_count = sum(1 for ch in answer if not ch.isalnum() and not ch.isspace())
    upper_count = sum(1 for ch in answer if ch.isalpha() and ch.isupper())
    alpha_count = max(1, sum(1 for ch in answer if ch.isalpha()))

    return {
        "unc_len_tokens": float(token_count),
        "unc_rare_token_ratio": rare_like / token_count,
        "unc_long_token_ratio": long_tokens / token_count,
        "unc_prompt_overlap_ratio": overlap / token_count,
        "unc_symbol_ratio": symbol_count / max(1, len(answer)),
        "unc_uppercase_ratio": upper_count / alpha_count,
    }


def _internal_features(prompt: str, answer: str) -> dict[str, float]:
    p_numbers = set(NUMBER_RE.findall(prompt))
    a_numbers = NUMBER_RE.findall(answer)
    p_years = set(YEAR_RE.findall(prompt))
    a_years = set(YEAR_RE.findall(answer))

    p_words = set(WORD_RE.findall(prompt))
    a_words = set(WORD_RE.findall(answer))

    new_words = a_words - p_words
    jaccard = len(a_words & p_words) / max(1, len(a_words | p_words))

    return {
        "int_number_count": float(len(a_numbers)),
        "int_number_novel_ratio": (
            sum(1 for n in a_numbers if n not in p_numbers) / max(1, len(a_numbers))
        ),
        "int_year_conflict": float(bool(a_years and p_years and not (a_years & p_years))),
        "int_new_word_ratio": len(new_words) / max(1, len(a_words)),
        "int_prompt_answer_jaccard": jaccard,
    }


def _optional_lm_internal_features(
    answer: str,
    *,
    lm_bundle: dict[str, Any] | None,
    device: str = "cpu",
) -> dict[str, float]:
    """Optional transformer-based internal features; safe fallback when deps absent.

    Returns zeros if model/tokenizer/torch are unavailable.
    """

    base = {
        "lm_hidden_norm": 0.0,
        "lm_hidden_var": 0.0,
        "lm_token_logprob_mean": 0.0,
        "lm_token_logprob_p10": 0.0,
    }
    if not lm_bundle:
        return base

    try:
        import torch
    except Exception:
        return base

    model = lm_bundle.get("model")
    tokenizer = lm_bundle.get("tokenizer")
    if model is None or tokenizer is None:
        return base

    if device == "cpu":
        model = model.to("cpu")
    encoded = tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    if device == "cpu":
        encoded = {k: v.to("cpu") for k, v in encoded.items()}
    else:
        encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        logits = out.logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    base["lm_hidden_norm"] = float(last_hidden.norm(dim=-1).mean().item())
    base["lm_hidden_var"] = float(last_hidden.var(dim=-1).mean().item())
    if token_lp.numel() > 0:
        base["lm_token_logprob_mean"] = float(token_lp.mean().item())
        base["lm_token_logprob_p10"] = float(torch.quantile(token_lp, 0.1).item())
    return base


def build_features(
    df: pd.DataFrame,
    *,
    device: str = "cpu",
    lm_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        prompt = preprocess(row["prompt"])
        answer = preprocess(row["model_answer"])

        acc = FeatureAccumulator()
        for name, value in _uncertainty_features(prompt, answer).items():
            acc.add(name, value)
        for name, value in _internal_features(prompt, answer).items():
            acc.add(name, value)
        for name, value in _optional_lm_internal_features(
            answer, lm_bundle=lm_bundle, device=device
        ).items():
            acc.add(name, value)

        rows.append(acc.to_dict())

    return pd.DataFrame(rows).fillna(0.0)
