"""Vocabulary-coverage and caption-length statistics."""

import numpy as np
import pandas as pd

from .tokenizer import CaptionTokenizer


def compute_text_stats(captions: list[str], tokenizer: CaptionTokenizer, split_name: str) -> dict:
    """Measure how well a split is covered by the (train-built) vocabulary.

    The OOV rate is the number that matters: it tells you what fraction of the
    words in this split the model literally cannot produce, and therefore puts
    a ceiling on achievable caption quality no matter how good the encoder is.
    """
    if not captions:
        return {}

    vocab_set = set(tokenizer.vocab)
    all_words: list[str] = []
    lengths: list[int] = []

    for caption in captions:
        words = tokenizer.clean_caption(caption).split()
        all_words.extend(words)
        lengths.append(len(words))

    in_vocab = sum(1 for w in all_words if w in vocab_set)
    coverage = in_vocab / len(all_words) * 100 if all_words else 0.0

    return {
        "split": split_name,
        "num_captions": len(captions),
        "total_words": len(all_words),
        "unique_words": len(set(all_words)),
        "vocab_coverage": coverage,
        "oov_percentage": 100.0 - coverage,
        "avg_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "lengths": lengths,
    }


def stats_table(stats: list[dict]) -> pd.DataFrame:
    """Format a list of stat dicts into a printable table (drops raw lengths)."""
    columns = [
        "split", "num_captions", "total_words", "unique_words",
        "vocab_coverage", "oov_percentage", "avg_length", "max_length",
    ]
    df = pd.DataFrame([s for s in stats if s])
    df["vocab_coverage"] = df["vocab_coverage"].round(2)
    df["oov_percentage"] = df["oov_percentage"].round(2)
    df["avg_length"] = df["avg_length"].round(1)
    return df[columns]
