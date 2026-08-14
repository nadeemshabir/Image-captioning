"""Model evaluation: aggregate metrics, qualitative examples, error slices."""

import random

import numpy as np
import torch

from ..training.loops import unpack_batch
from ..training.memory import cleanup
from .metrics import (
    aggregate,
    compute_bleu4,
    compute_meteor,
    has_repetition,
    tokens_from_indices,
)


def _prepare_for_inference(model, device):
    """Move a model to the device in fp32.

    The dataset emits fp16 images to save bandwidth, but generation runs
    outside ``autocast``, so model and input must agree on dtype or the first
    matmul raises. Standardising on fp32 at inference is the simpler half of
    that contract to enforce.
    """
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.float()
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, model_name, device,
                   max_batches: int = 100, max_gen_length: int = 25) -> dict:
    """Score a model on BLEU-4, METEOR, caption length and repetition rate."""
    model = _prepare_for_inference(model, device)

    generated_tokens: list[list[str]] = []
    reference_tokens: list[list[list[str]]] = []
    lengths: list[int] = []
    repetitions = 0

    print(f"Evaluating {model_name}...")

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images, captions, _ = unpack_batch(batch)
        images = images.to(device, dtype=torch.float32)

        try:
            generated = model.generate_caption(images, tokenizer, max_length=max_gen_length)
        except RuntimeError as exc:
            print(f"  batch {i} failed ({str(exc)[:50]}...) - skipping")
            continue

        for j, caption in enumerate(generated):
            words = caption.split() if isinstance(caption, str) else list(caption)
            generated_tokens.append(words)
            lengths.append(len(words))
            if has_repetition(words):
                repetitions += 1
            # BLEU wants a *list of* references, hence the extra nesting.
            reference_tokens.append([tokens_from_indices(captions[j], tokenizer, keep_unk=True)])

    model.to("cpu")
    cleanup()

    bleu = [compute_bleu4(g, r) for g, r in zip(generated_tokens, reference_tokens)]
    meteor = [compute_meteor(g, r) for g, r in zip(generated_tokens, reference_tokens)]
    bleu_mean, bleu_std = aggregate(bleu)
    meteor_mean, meteor_std = aggregate(meteor)

    total = max(1, len(generated_tokens))
    return {
        "model_name": model_name,
        "bleu4_mean": bleu_mean,
        "bleu4_std": bleu_std,
        "meteor_mean": meteor_mean,
        "meteor_std": meteor_std,
        "caption_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "caption_length_std": float(np.std(lengths)) if lengths else 0.0,
        "repetition_percentage": repetitions / total * 100,
        "total_samples": len(generated_tokens),
        "samples": {
            "generated": [" ".join(t) for t in generated_tokens[:5]],
            "references": [" ".join(r[0]) for r in reference_tokens[:5]],
        },
    }


@torch.no_grad()
def success_failure_examples(model, dataset, tokenizer, device,
                             num_samples: int = 100, top_k: int = 5):
    """Return the best- and worst-scoring captions from a random sample.

    Reading the extremes is the fastest way to tell *how* a model is failing:
    a low mean BLEU caused by fluent-but-generic captions is a very different
    problem from one caused by degenerate repetition.
    """
    model = _prepare_for_inference(model, device)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    scored = []

    for idx in indices:
        try:
            image, caption, _ = dataset[idx]
            image_batch = image.unsqueeze(0).to(device, dtype=torch.float32)
            generated = model.generate_caption(image_batch, tokenizer, max_length=20)

            gen_words = generated[0].split() if isinstance(generated[0], str) else list(generated[0])
            ref_words = tokens_from_indices(caption, tokenizer, keep_unk=True)

            scored.append({
                "index": idx,
                "bleu": compute_bleu4(gen_words, [ref_words]),
                "generated": " ".join(gen_words),
                "reference": " ".join(ref_words),
            })
        except RuntimeError:
            continue

    model.to("cpu")
    scored.sort(key=lambda r: r["bleu"], reverse=True)
    return scored[:top_k], scored[-top_k:]


@torch.no_grad()
def error_slice_analysis(model, dataset, tokenizer, device,
                         sample_size: int = 200, per_slice: int = 20) -> dict:
    """Score short (<=8 words) and long (>=16 words) references separately.

    A single mean hides length bias. If a model does much worse on long
    references it is usually stopping too early, which is a decoding problem
    rather than a perception one.
    """
    model = _prepare_for_inference(model, device)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    slices: dict[str, list] = {"short_captions": [], "long_captions": []}
    for idx in indices:
        try:
            image, caption, _ = dataset[idx]
            ref_words = tokens_from_indices(caption, tokenizer, keep_unk=True)
            if len(ref_words) <= 8:
                slices["short_captions"].append((image, ref_words))
            elif len(ref_words) >= 16:
                slices["long_captions"].append((image, ref_words))
        except Exception:
            continue

    results = {}
    for name, items in slices.items():
        if not items:
            continue
        scores = []
        for image, ref_words in items[:per_slice]:
            try:
                image_batch = image.unsqueeze(0).to(device, dtype=torch.float32)
                generated = model.generate_caption(image_batch, tokenizer, max_length=20)
                gen_words = generated[0].split() if isinstance(generated[0], str) else list(generated[0])
                scores.append(compute_bleu4(gen_words, [ref_words]))
            except RuntimeError:
                continue

        mean, _ = aggregate(scores)
        results[name] = {
            "mean_bleu": mean,
            "sample_count": len(items),
            "evaluated_count": len(scores),
        }

    model.to("cpu")
    cleanup()
    return results
