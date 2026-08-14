"""Evaluate trained checkpoints on BLEU-4, METEOR, length and repetition.

    python scripts/evaluate.py --data-root data --checkpoint-dir checkpoints
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
from rsicd_captioner.evaluation import (
    ensure_nltk_data,
    error_slice_analysis,
    evaluate_model,
    success_failure_examples,
)
from rsicd_captioner.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS),
                        choices=list(MODEL_VARIANTS))
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--qualitative", action="store_true",
                        help="also print best/worst captions and length slices")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    return parser.parse_args()


def load_checkpoint(name: str, vocab_size: int, config: Config, checkpoint_dir: Path, device):
    """Rebuild a model and load its weights, if a checkpoint exists."""
    encoder_name, decoder_type = MODEL_VARIANTS[name]
    # pretrained=False: the checkpoint supplies every weight, so downloading
    # ImageNet weights first would just be thrown away.
    config.model.pretrained = False
    model = build_model(vocab_size, encoder_name, decoder_type, config.model)

    path = checkpoint_dir / f"{name.replace('+', '_').lower()}.pt"
    if not path.exists():
        print(f"  no checkpoint at {path} - skipping {name}")
        return None

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    config = Config()
    if args.data_root:
        config.data.root = args.data_root

    ensure_nltk_data()
    device = get_device()
    print(f"Device: {device}")

    train_df, valid_df, test_df = load_splits(config.data)
    tokenizer = CaptionTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length,
        min_word_freq=config.tokenizer.min_word_freq,
    )
    # The vocabulary must be rebuilt from train exactly as it was at training
    # time, or the saved output layer will not line up with the word indices.
    tokenizer.build_vocabulary(extract_all_captions(train_df))

    preprocessor = ImagePreprocessor(
        target_size=config.data.image_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
    )
    eval_df = valid_df if args.split == "valid" else test_df
    dataset = RSICDDataset(eval_df, tokenizer, preprocessor,
                           config.data.max_captions_per_image)

    subset = Subset(dataset, range(0, min(args.samples, len(dataset))))
    loader = DataLoader(subset, batch_size=config.evaluation.eval_batch_size,
                        shuffle=False, num_workers=0)

    results = {}
    for name in args.models:
        model = load_checkpoint(name, len(tokenizer), config, args.checkpoint_dir, device)
        if model is None:
            continue

        results[name] = evaluate_model(
            model, loader, tokenizer, name, device,
            max_gen_length=config.evaluation.max_gen_length,
        )

        r = results[name]
        print(f"\n{name}")
        print(f"  BLEU-4     : {r['bleu4_mean']:.4f} +/- {r['bleu4_std']:.4f}")
        print(f"  METEOR     : {r['meteor_mean']:.4f} +/- {r['meteor_std']:.4f}")
        print(f"  length     : {r['caption_length_mean']:.1f} +/- {r['caption_length_std']:.1f} words")
        print(f"  repetition : {r['repetition_percentage']:.1f}%")
        for gen, ref in zip(r["samples"]["generated"][:2], r["samples"]["references"][:2]):
            print(f"    gen: {gen}")
            print(f"    ref: {ref}")

        if args.qualitative:
            best, worst = success_failure_examples(model, dataset, tokenizer, device)
            print("\n  best captions:")
            for ex in best[:3]:
                print(f"    [{ex['bleu']:.4f}] {ex['generated']}")
            print("  worst captions:")
            for ex in worst[:3]:
                print(f"    [{ex['bleu']:.4f}] {ex['generated']}")

            slices = error_slice_analysis(model, dataset, tokenizer, device)
            results[name]["error_slices"] = slices
            for slice_name, s in slices.items():
                print(f"  {slice_name}: BLEU-4 {s['mean_bleu']:.4f} "
                      f"({s['evaluated_count']} evaluated)")

        del model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
