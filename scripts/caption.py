"""Caption a single image with one or more trained models.

    python scripts/caption.py path/to/image.jpg
    python scripts/caption.py image.jpg --models ResNet+Transformer --gradcam
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
from PIL import Image

from rsicd_captioner.config import MODEL_VARIANTS, Config, get_device
from rsicd_captioner.data import (
    CaptionTokenizer,
    ImagePreprocessor,
    extract_all_captions,
    load_splits,
)
from rsicd_captioner.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to an image file")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="needed to rebuild the training vocabulary")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS),
                        choices=list(MODEL_VARIANTS))
    parser.add_argument("--max-length", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    config = Config()
    if args.data_root:
        config.data.root = args.data_root
    device = get_device()

    # The vocabulary is not stored in the checkpoint, so it is rebuilt from the
    # training captions exactly as it was at training time.
    train_df, _, _ = load_splits(config.data)
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
    image = Image.open(args.image).convert("RGB")
    array = preprocessor.preprocess_image(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device)

    config.model.pretrained = False
    print(f"\nCaptions for {args.image.name}")
    print("=" * 60)

    for name in args.models:
        path = args.checkpoint_dir / f"{name.replace('+', '_').lower()}.pt"
        if not path.exists():
            print(f"{name:<24} (no checkpoint)")
            continue

        encoder_name, decoder_type = MODEL_VARIANTS[name]
        model = build_model(len(tokenizer), encoder_name, decoder_type, config.model)
        model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
        model.to(device).float().eval()

        with torch.no_grad():
            caption = model.generate_caption(tensor, tokenizer, max_length=args.max_length)[0]

        print(f"{name:<24} {caption}")
        model.to("cpu")
        del model


if __name__ == "__main__":
    main()
