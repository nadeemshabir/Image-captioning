"""Run the ablation experiments.

    python scripts/run_experiments.py --experiments all
    python scripts/run_experiments.py --experiments backbones length_penalty

Experiments 1-3 and 7 reuse the trained checkpoints; 4-6 train small purpose-built
variants from scratch on a data subset, so they do not need checkpoints at all.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
from torch.utils.data import DataLoader, Subset

from rsicd_captioner.config import MODEL_VARIANTS, Config, get_device
from rsicd_captioner.data import (
    CaptionTokenizer,
    ImagePreprocessor,
    RSICDDataset,
    extract_all_captions,
    load_splits,
)
from rsicd_captioner.evaluation import ensure_nltk_data
from rsicd_captioner.experiments import (
    backbones,
    dropout_placement,
    length_penalty,
    memory_tokens,
    rotation,
    vision_text_interface,
)
from rsicd_captioner.models import build_model

# Which experiments need trained weights loaded.
NEEDS_CHECKPOINTS = {"backbones", "rotation", "interface", "length_penalty"}
ALL_EXPERIMENTS = [
    "backbones", "rotation", "interface",
    "lstm_interface", "memory_tokens", "dropout", "length_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=ALL_EXPERIMENTS + ["all"])
    parser.add_argument("--epochs", type=int, default=3,
                        help="epochs for the from-scratch experiments (4-6)")
    parser.add_argument("--output", type=Path, default=Path("results/experiments.json"))
    return parser.parse_args()


def load_models(names, vocab_size, config, checkpoint_dir):
    """Load whichever checkpoints are available."""
    config.model.pretrained = False
    models = {}
    for name in names:
        path = checkpoint_dir / f"{name.replace('+', '_').lower()}.pt"
        if not path.exists():
            continue
        encoder_name, decoder_type = MODEL_VARIANTS[name]
        model = build_model(vocab_size, encoder_name, decoder_type, config.model)
        model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
        models[name] = model
    return models


def main() -> None:
    args = parse_args()
    selected = ALL_EXPERIMENTS if "all" in args.experiments else args.experiments

    config = Config()
    if args.data_root:
        config.data.root = args.data_root

    ensure_nltk_data()
    device = get_device()
    torch.manual_seed(config.seed)
    print(f"Device: {device}")

    train_df, valid_df, _ = load_splits(config.data)
    tokenizer = CaptionTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length,
        min_word_freq=config.tokenizer.min_word_freq,
    )
    tokenizer.build_vocabulary(extract_all_captions(train_df))

    preprocessor = ImagePreprocessor(
        target_size=config.data.image_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
    )
    train_dataset = RSICDDataset(train_df, tokenizer, preprocessor,
                                 config.data.max_captions_per_image)
    val_dataset = RSICDDataset(valid_df, tokenizer, preprocessor,
                               config.data.max_captions_per_image)

    models = {}
    if NEEDS_CHECKPOINTS & set(selected):
        models = load_models(list(MODEL_VARIANTS), len(tokenizer), config, args.checkpoint_dir)
        if not models:
            print(f"No checkpoints found in {args.checkpoint_dir}. "
                  "Run scripts/train.py first, or pick only experiments 4-6 "
                  "(lstm_interface, memory_tokens, dropout).")

    results = {}

    if "backbones" in selected and models:
        subset = Subset(val_dataset, range(0, min(500, len(val_dataset))))
        loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=0)
        results["backbone_comparison"] = backbones.run(models, loader, tokenizer, device)

    if "rotation" in selected and models:
        # One model per decoder family is enough to see the pattern.
        chosen = {k: v for k, v in models.items()
                  if k in ("ResNet+LSTM", "MobileNet+Transformer")}
        results["rotation_invariance"] = {
            name: {"consistency_score": r["consistency_score"]}
            for name, r in rotation.run(chosen or models, val_dataset, tokenizer, device).items()
        }

    if "interface" in selected and models:
        results["vision_text_interface"] = vision_text_interface.run_interface_comparison(
            models, val_dataset, tokenizer, device
        )

    if "lstm_interface" in selected:
        results["lstm_interface"] = vision_text_interface.run_lstm_interface(
            train_dataset, val_dataset, tokenizer, device, epochs=args.epochs
        )

    if "memory_tokens" in selected:
        results["transformer_memory"] = memory_tokens.run(
            train_dataset, val_dataset, tokenizer, device, epochs=args.epochs
        )

    if "dropout" in selected:
        results["dropout_placement"] = dropout_placement.run(
            train_dataset, val_dataset, tokenizer, device, epochs=args.epochs
        )

    if "length_penalty" in selected and models:
        raw = length_penalty.run(models, val_dataset, tokenizer, device,
                                 penalties=config.evaluation.length_penalties)
        # JSON object keys must be strings; float penalties are not.
        results["length_penalty"] = {
            model: {str(p): v for p, v in sweep.items()} for model, sweep in raw.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
