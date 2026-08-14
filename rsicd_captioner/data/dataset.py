"""Dataset and caption-parsing helpers for RSICD."""

import ast
import io
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import ImagePreprocessor
from .tokenizer import CaptionTokenizer


def parse_caption_list(caption_field) -> list[str]:
    """Extract the individual captions from one CSV cell.

    The ``captions`` column holds a stringified list whose elements are quoted
    with single quotes, e.g. ``"['a road .' 'a wide road .']"``. A regex is
    more forgiving here than ``ast.literal_eval``: some rows are missing the
    comma separators, which would make a strict parse fail on otherwise
    perfectly usable text.
    """
    try:
        if pd.isna(caption_field):
            return []
    except (TypeError, ValueError):
        pass  # non-scalar input; fall through to the regex

    captions = re.findall(r"'([^']*)'", str(caption_field))
    return [c.strip() for c in captions if c.strip()]


def extract_all_captions(df: pd.DataFrame) -> list[str]:
    """Flatten every caption in a split into one list (used to build the vocab)."""
    captions: list[str] = []
    for _, row in df.iterrows():
        captions.extend(parse_caption_list(row["captions"]))
    return captions


class RSICDDataset(Dataset):
    """Caption-level dataset over the RSICD CSVs.

    One item = one (image, caption) pair, so an image with 5 captions yields 5
    items. The image itself is decoded lazily.

    **Why lazy decoding matters.** The obvious implementation decodes and
    preprocesses every image in ``__init__`` and stores the resulting arrays.
    For ~10.9k images at 224x224x3 float32 that is roughly 6 GB of resident
    memory before training even starts - and because each image is repeated
    once per caption, the naive version actually stores it five times over.
    Keeping only row indices and pre-tokenised captions brings the dataset's
    own footprint down to tens of megabytes, and lets DataLoader workers fork
    cheaply instead of each inheriting a multi-gigabyte copy.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: CaptionTokenizer,
        preprocessor: ImagePreprocessor,
        max_captions_per_image: int = 5,
        verbose: bool = True,
    ):
        self.df = df.reset_index(drop=True)  # keep a reference, never a copy of the bytes
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

        # Two parallel lists: which dataframe row an item points at, and the
        # already-padded token ids for that item's caption.
        self.indices: list[int] = []
        self.caption_tokens: list[list[int]] = []

        for row_idx, row in self.df.iterrows():
            try:
                captions = parse_caption_list(row["captions"])
                for caption in captions[:max_captions_per_image]:
                    encoded = tokenizer.encode_caption(caption)
                    self.caption_tokens.append(tokenizer.pad_sequence(encoded))
                    self.indices.append(row_idx)
            except Exception:
                # A handful of rows have malformed caption fields; skipping a
                # few out of ~10.9k is preferable to aborting the whole build.
                continue

        if verbose:
            print(f"Dataset ready: {len(self.caption_tokens)} caption items "
                  f"from {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.caption_tokens)

    def __getitem__(self, idx: int):
        """Decode one image on demand and pair it with its caption tensor."""
        row = self.df.iloc[self.indices[idx]]
        pad_idx = self.tokenizer.word_to_idx[CaptionTokenizer.PAD_TOKEN]

        try:
            # Images live in the CSV as raw JPEG bytes wrapped in a dict literal.
            image_dict = ast.literal_eval(row["image"])
            image = Image.open(io.BytesIO(image_dict["bytes"]))
            processed = self.preprocessor.preprocess_image(image)

            # Store as fp16 when a GPU is present: it halves the bytes moved
            # through pinned memory and over PCIe, and the training loop runs
            # under autocast anyway. On CPU, fp32 avoids slow fp16 emulation.
            dtype = np.float16 if torch.cuda.is_available() else np.float32
            image_tensor = torch.from_numpy(processed.astype(dtype)).permute(2, 0, 1)

            caption_tensor = torch.LongTensor(self.caption_tokens[idx])
            length = sum(1 for t in self.caption_tokens[idx] if t != pad_idx)
            return image_tensor, caption_tensor, length

        except Exception:
            # A corrupt image must not kill an epoch. Return a blank sample the
            # loss can still consume; these are rare enough not to move metrics.
            dummy_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            dummy_image = torch.zeros(3, *self.preprocessor.target_size, dtype=dummy_dtype)

            bos = self.tokenizer.word_to_idx[CaptionTokenizer.BOS_TOKEN]
            eos = self.tokenizer.word_to_idx[CaptionTokenizer.EOS_TOKEN]
            dummy_caption = torch.LongTensor(
                [bos, eos] + [pad_idx] * (self.tokenizer.max_length - 2)
            )
            return dummy_image, dummy_caption, 2


def load_splits(data_config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three CSV splits described by a :class:`DataConfig`."""
    train_df = pd.read_csv(data_config.train_path)
    valid_df = pd.read_csv(data_config.valid_path)
    test_df = pd.read_csv(data_config.test_path)
    return train_df, valid_df, test_df
