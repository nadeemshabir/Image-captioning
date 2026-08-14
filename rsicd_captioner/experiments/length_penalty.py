"""Experiment 7 - a gentle length penalty during greedy decoding.

Greedy decoding tends to stop early: <eos> is a high-frequency token, so it
often wins the argmax before the caption has said anything specific. Subtracting
a small, step-dependent amount from the <eos> logit delays that decision without
forbidding it outright.

An important caveat on interpreting the results: the penalty only changes the
output if <eos> was close to winning. Where a model's argmax is decisive, the
generated sequence is identical at every penalty value, and identical output
means identical BLEU. Flat rows across the sweep are a real finding about the
model's confidence, not a broken experiment.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import aggregate, compute_bleu4, tokens_from_indices
from ..training.loops import unpack_batch
from ..training.memory import cleanup


@torch.no_grad()
def greedy_decode_with_penalty(model, images, tokenizer, max_length: int = 25,
                               length_penalty: float = 0.0, warmup_steps: int = 5,
                               scale: float = 0.1) -> torch.Tensor:
    """Greedy decode, damping the <eos> logit after ``warmup_steps`` tokens.

    The penalty grows linearly with how far past the warm-up we are, so short
    captions are nudged and long ones are left alone - a hard constraint on
    minimum length would instead produce padding-like filler.
    """
    model.eval()
    batch_size = images.size(0)
    device = images.device
    bos = tokenizer.word_to_idx["<bos>"]
    eos = tokenizer.word_to_idx["<eos>"]

    generated = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)

    for step in range(max_length):
        # Feed the whole prefix back through the model, matching how it was
        # trained (teacher forcing over a full sequence).
        outputs = model(images, generated)
        logits = outputs[:, -1, :].clone()

        if length_penalty > 0 and step > warmup_steps:
            logits[:, eos] -= length_penalty * (step - warmup_steps) * scale

        next_tokens = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)

        if (next_tokens == eos).all():
            break

    return generated


def compare_length_penalties(model, dataloader, tokenizer, model_name, device,
                             penalties=(0.0, 0.5, 1.0, 2.0), max_batches: int = 10) -> dict:
    """Sweep penalty strengths on a fixed set of batches."""
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.float()

    # Materialise the batches once so every penalty sees identical inputs.
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images, captions, _ = unpack_batch(batch)
        batches.append((images.to(device, dtype=torch.float32), captions))

    results = {}
    for penalty in penalties:
        generated_all, references_all, lengths = [], [], []

        for images, captions in batches:
            try:
                generated = greedy_decode_with_penalty(
                    model, images, tokenizer, max_length=25, length_penalty=penalty
                )
            except RuntimeError as exc:
                print(f"  generation failed ({str(exc)[:40]}...)")
                continue

            for j in range(generated.size(0)):
                gen_tokens = tokens_from_indices(generated[j], tokenizer)
                ref_tokens = tokens_from_indices(captions[j], tokenizer)
                if gen_tokens and ref_tokens:
                    generated_all.append(gen_tokens)
                    references_all.append([ref_tokens])
                    lengths.append(len(gen_tokens))

        scores = [compute_bleu4(g, r) for g, r in zip(generated_all, references_all)]
        mean, _ = aggregate(scores)

        results[penalty] = {
            "bleu4_mean": mean,
            "bleu4_max": float(np.max(scores)) if scores else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "samples_generated": len(generated_all),
            "sample_captions": [" ".join(t) for t in generated_all[:3]],
        }
        print(f"  penalty {penalty}: BLEU-4 {mean:.4f}, avg length {results[penalty]['avg_length']:.1f}")

    model.to("cpu")
    cleanup()
    return results


def run(models: dict, dataset, tokenizer, device, num_samples: int = 50,
        batch_size: int = 4, penalties=(0.0, 0.5, 1.0, 2.0)) -> dict:
    """Experiment 7 entry point - sweeps the penalty for every model."""
    print("\nExperiment 7: length penalty during greedy decoding")
    print("=" * 60)

    from torch.utils.data import Subset
    subset = Subset(dataset, range(0, min(num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    for name, model in models.items():
        print(f"\n{name}")
        results[name] = compare_length_penalties(
            model, loader, tokenizer, name, device, penalties=penalties
        )
    return results
