"""Train the captioning models.

    python scripts/train.py --data-root data --epochs 5
    python scripts/train.py --models ResNet+Transformer --batch-size 128

Models are trained one at a time and moved back to CPU afterwards, so peak
VRAM is set by the largest single model rather than by all four together.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsicd_captioner.config import MODEL_VARIANTS, Config, get_device
from rsicd_captioner.data import (
    CaptionTokenizer,
    ImagePreprocessor,
    RSICDDataset,
    extract_all_captions,
    load_splits,
)
from rsicd_captioner.models import build_model
from rsicd_captioner.training import report_memory, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS),
                        choices=list(MODEL_VARIANTS),
                        help="which variants to train (default: all four)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--max-captions", type=int, default=None,
                        help="captions used per image (1-5); lower = smaller epoch")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results", type=Path, default=Path("results/training_history.json"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    if args.data_root:
        config.data.root = args.data_root
    if args.max_captions:
        config.data.max_captions_per_image = args.max_captions
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.accumulation_steps:
        config.training.accumulation_steps = args.accumulation_steps
    if args.no_amp:
        config.training.mixed_precision = False
    config.training.checkpoint_dir = args.checkpoint_dir
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    torch.manual_seed(config.seed)

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    report_memory()

    # ---------------- data ----------------
    train_df, valid_df, _ = load_splits(config.data)
    train_captions = extract_all_captions(train_df)

    tokenizer = CaptionTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length,
        min_word_freq=config.tokenizer.min_word_freq,
    )
    tokenizer.build_vocabulary(train_captions)
    print(f"Vocabulary: {len(tokenizer)} words")

    preprocessor = ImagePreprocessor(
        target_size=config.data.image_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
    )
    train_dataset = RSICDDataset(train_df, tokenizer, preprocessor,
                                 config.data.max_captions_per_image)
    val_dataset = RSICDDataset(valid_df, tokenizer, preprocessor,
                               config.data.max_captions_per_image)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers, pin_memory=pin,
        persistent_workers=False,  # workers would otherwise hold the dataframe alive
        drop_last=True,            # keeps every step the same shape for cuDNN autotuning
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, pin_memory=pin,
        persistent_workers=False, drop_last=False,
    )

    # Padding contributes no information, so it must not contribute loss either.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_idx[CaptionTokenizer.PAD_TOKEN])

    # ---------------- train ----------------
    histories = {}
    for name in args.models:
        encoder_name, decoder_type = MODEL_VARIANTS[name]
        model = build_model(len(tokenizer), encoder_name, decoder_type, config.model)
        print(f"\n{name}: {model.parameter_count() / 1e6:.1f}M parameters")

        lr = (config.training.transformer_lr if decoder_type == "transformer"
              else config.training.lstm_lr)

        histories[name] = train_model(
            model, name, train_loader, val_loader, device, criterion, lr,
            config.training, checkpoint_dir=config.training.checkpoint_dir,
        )
        del model

    # ---------------- summary ----------------
    print("\nTraining summary")
    print("=" * 60)
    for name, h in histories.items():
        print(f"{name}: best val {h['best_val_loss']:.4f} (epoch {h['best_epoch']}), "
              f"final train {h['final_train_loss']:.4f}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print(f"\nHistory saved to {args.results}")


if __name__ == "__main__":
    main()
