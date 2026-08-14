"""BLEU / METEOR scoring and token-handling helpers.

Most of the subtlety in caption evaluation is not the metric itself but
getting the inputs into the shape NLTK expects. Both ``sentence_bleu`` and
``meteor_score`` want **lists of word strings**, never raw index tensors and
never a joined sentence - passing a string makes NLTK iterate over characters
and silently return a meaningless score rather than raising.
"""

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score

# Structural tokens are stripped before scoring: they are identical in every
# caption, so leaving them in would inflate n-gram overlap for free.
STRUCTURAL_TOKENS = {"<bos>", "<eos>", "<pad>", "<unk>"}


def ensure_nltk_data() -> None:
    """Make sure the corpora METEOR depends on are present."""
    import nltk
    for resource, path in [("punkt", "tokenizers/punkt"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def tokens_from_indices(indices, tokenizer, keep_unk: bool = False) -> list[str]:
    """Convert an index sequence (list or tensor) into a list of word strings."""
    skip = STRUCTURAL_TOKENS - {"<unk>"} if keep_unk else STRUCTURAL_TOKENS

    words = []
    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        word = tokenizer.idx_to_word.get(int(idx), "<unk>")
        if word not in skip:
            words.append(word)
    return words


def as_token_list(caption) -> list[str]:
    """Normalise a caption (string or list) to a list of word strings."""
    if isinstance(caption, str):
        return caption.split()
    return list(caption)


def compute_bleu4(hypothesis, references, smooth: bool = True) -> float:
    """BLEU-4 for one caption against one or more references.

    Smoothing matters here: BLEU multiplies its n-gram precisions, so a single
    caption with no matching 4-gram scores exactly 0 and drags the mean down
    regardless of how good its unigrams were. Smoothing method 1 adds a small
    epsilon to zero counts, which keeps short-caption scores informative.
    """
    hyp = as_token_list(hypothesis)
    if not hyp or not references:
        return 0.0
    try:
        kwargs = {"weights": (0.25, 0.25, 0.25, 0.25)}
        if smooth:
            kwargs["smoothing_function"] = SmoothingFunction().method1
        return float(sentence_bleu(references, hyp, **kwargs))
    except (ZeroDivisionError, ValueError):
        return 0.0


def compute_meteor(hypothesis, references) -> float:
    """METEOR for one caption. Both arguments must be tokenised word lists."""
    hyp = as_token_list(hypothesis)
    if not hyp or not references:
        return 0.0
    try:
        return float(meteor_score(references, hyp))
    except (ValueError, LookupError):
        return 0.0


def has_repetition(tokens: list[str], run_length: int = 3) -> bool:
    """True if the caption contains ``run_length`` identical consecutive words.

    Degenerate repetition ("are are are") is the classic failure mode of a
    greedy decoder that has not learned a strong enough language model, and it
    is invisible to BLEU averages - worth counting separately.
    """
    if len(tokens) < run_length:
        return False
    for i in range(len(tokens) - run_length + 1):
        if len(set(tokens[i: i + run_length])) == 1:
            return True
    return False


def aggregate(scores: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)``, or ``(0, 0)`` for an empty list."""
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))
