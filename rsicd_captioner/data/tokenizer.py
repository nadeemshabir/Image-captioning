"""Word-level tokenizer for RSICD captions."""

import re
from collections import Counter
from typing import Iterable, Sequence


class CaptionTokenizer:
    """Builds a word-level vocabulary and converts captions to index sequences.

    Design notes:

    * The vocabulary is built from the **training split only**. Letting
      validation or test words leak into the vocabulary would inflate coverage
      numbers and quietly overstate how well the model generalises.
    * The four special tokens occupy indices 0-3 by construction, so
      ``pad_idx == 0`` can be relied on by ``nn.Embedding(padding_idx=...)``
      and by the loss function's ``ignore_index``.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab_size: int = 10_000, max_length: int = 24, min_word_freq: int = 2):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_word_freq = min_word_freq

        # Populated by build_vocabulary().
        self.vocab: list[str] | None = None
        self.word_to_idx: dict[str, int] | None = None
        self.idx_to_word: dict[int, str] | None = None

    # ------------------------------------------------------------------ #
    # Text normalisation
    # ------------------------------------------------------------------ #

    def clean_caption(self, caption: str) -> str:
        """Lowercase, collapse whitespace, and split punctuation into its own token.

        Punctuation is separated rather than stripped because a trailing "."
        is a genuine signal that the sentence is finished - useful for a model
        that has to learn when to emit <eos>.
        """
        caption = caption.lower().strip()
        caption = re.sub(r"\s+", " ", caption)
        caption = re.sub(r"([.!?,:;])", r" \1 ", caption)
        return re.sub(r"\s+", " ", caption).strip()

    # ------------------------------------------------------------------ #
    # Vocabulary
    # ------------------------------------------------------------------ #

    def build_vocabulary(self, captions: Iterable[str]) -> None:
        """Count words across the training captions and keep the most frequent.

        Words seen fewer than ``min_word_freq`` times are dropped and will map
        to <unk>. Keeping singletons would bloat the softmax with words the
        model can never learn to place correctly.
        """
        all_words: list[str] = []
        for caption in captions:
            all_words.extend(self.clean_caption(caption).split())

        word_freq = Counter(all_words)
        frequent = [w for w, freq in word_freq.items() if freq >= self.min_word_freq]
        # Sort by frequency so that truncating to vocab_size drops the rarest.
        ordered = sorted(frequent, key=lambda w: word_freq[w], reverse=True)
        kept = ordered[: self.vocab_size - 4]  # leave room for the 4 specials

        self.vocab = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN] + kept
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        # Note the direction here: this is the *inverse* map, so the index is
        # the key. Writing {word: idx for idx, word in ...} would silently
        # rebuild the forward map and make every decode return garbage.
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #

    def encode_caption(self, caption: str) -> list[int]:
        """Turn a caption into ``[<bos>, w1, ..., wn, <eos>]`` index form."""
        self._require_vocab()
        tokens = self.clean_caption(caption).split()

        indices = [self.word_to_idx[self.BOS_TOKEN]]
        for token in tokens:
            indices.append(self.word_to_idx.get(token, self.word_to_idx[self.UNK_TOKEN]))
        indices.append(self.word_to_idx[self.EOS_TOKEN])

        # Truncate over-long captions but always keep <eos> as the final token,
        # otherwise the model never sees a stop signal for those samples.
        if len(indices) > self.max_length:
            indices = indices[: self.max_length - 1] + [self.word_to_idx[self.EOS_TOKEN]]
        return indices

    def pad_sequence(self, indices: Sequence[int]) -> list[int]:
        """Right-pad (or truncate) an index sequence to exactly ``max_length``."""
        self._require_vocab()
        indices = list(indices)
        if len(indices) >= self.max_length:
            return indices[: self.max_length]
        pad = [self.word_to_idx[self.PAD_TOKEN]] * (self.max_length - len(indices))
        return indices + pad

    def decode_caption(self, indices: Iterable[int]) -> str:
        """Turn indices back into a readable sentence, dropping structural tokens."""
        self._require_vocab()
        skip = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        words = []
        for idx in indices:
            token = self.idx_to_word.get(int(idx))
            if token is not None and token not in skip:
                words.append(token)
        return " ".join(words)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        self._require_vocab()
        return len(self.vocab)

    def _require_vocab(self) -> None:
        if self.vocab is None:
            raise RuntimeError("build_vocabulary() must be called before using the tokenizer")
