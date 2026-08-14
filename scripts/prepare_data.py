"""Build the vocabulary and report train/val/test text statistics.

    python scripts/prepare_data.py --data-root data --output results/text_stats.csv
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from rsicd_captioner.config import Config
from rsicd_captioner.data import (
    CaptionTokenizer,
    compute_text_stats,
    extract_all_captions,
    load_splits,
    stats_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="directory holding train.csv / valid.csv / test.csv")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-word-freq", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/text_stats.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.data_root:
        config.data.root = args.data_root
    if args.vocab_size:
        config.tokenizer.vocab_size = args.vocab_size
    if args.min_word_freq:
        config.tokenizer.min_word_freq = args.min_word_freq
    if args.max_length:
        config.tokenizer.max_length = args.max_length

    train_df, valid_df, test_df = load_splits(config.data)
    print(f"Images  - train {len(train_df)}, val {len(valid_df)}, test {len(test_df)}")

    train_captions = extract_all_captions(train_df)
    val_captions = extract_all_captions(valid_df)
    test_captions = extract_all_captions(test_df)
    print(f"Captions - train {len(train_captions)}, val {len(val_captions)}, "
          f"test {len(test_captions)}")

    # Train split only: see CaptionTokenizer for why this matters.
    tokenizer = CaptionTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length,
        min_word_freq=config.tokenizer.min_word_freq,
    )
    tokenizer.build_vocabulary(train_captions)
    print(f"\nVocabulary: {len(tokenizer)} words "
          f"(cap {config.tokenizer.vocab_size}, min freq {config.tokenizer.min_word_freq})")
    print(f"Most frequent: {tokenizer.vocab[4:14]}")

    stats = [
        compute_text_stats(train_captions, tokenizer, "Train"),
        compute_text_stats(val_captions, tokenizer, "Validation"),
        compute_text_stats(test_captions, tokenizer, "Test"),
    ]
    table = stats_table(stats)
    print("\nText statistics")
    print("=" * 80)
    print(table.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
