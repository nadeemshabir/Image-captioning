"""Experiment 1 - backbone comparison: ResNet-18 vs MobileNet-v2.

Question: what does the ~4x smaller backbone actually cost in caption quality,
and what does it buy in speed and memory?
"""

import time

import numpy as np
import torch

from ..evaluation.metrics import aggregate, compute_bleu4, compute_meteor, tokens_from_indices
from ..training.loops import unpack_batch
from ..training.memory import cleanup


@torch.no_grad()
def measure_model_performance(model, dataloader, tokenizer, model_name, device,
                              max_batches: int = 50, max_gen_length: int = 20) -> dict:
    """Time generation and score quality for one model."""
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.float()
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1e6
    else:
        memory_before = 0.0

    generated_all: list[str] = []
    references_all: list[list[list[str]]] = []
    times: list[float] = []

    print(f"Measuring {model_name}...")

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images, captions, _ = unpack_batch(batch)
        images = images.to(device, dtype=torch.float32)

        # Synchronise around the timed region: CUDA kernels are launched
        # asynchronously, so without this we would be timing queue submission
        # rather than the work itself.
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        generated = model.generate_caption(images, tokenizer, max_length=max_gen_length)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        for j, caption in enumerate(generated):
            generated_all.append(caption)
            references_all.append([tokens_from_indices(captions[j], tokenizer, keep_unk=True)])

    memory_after = torch.cuda.memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    model.to("cpu")
    cleanup()

    bleu = [compute_bleu4(g, r) for g, r in zip(generated_all, references_all)]
    meteor = [compute_meteor(g, r) for g, r in zip(generated_all, references_all)]
    bleu_mean, _ = aggregate(bleu)
    meteor_mean, _ = aggregate(meteor)

    return {
        "model_name": model_name,
        "avg_inference_time_ms": float(np.mean(times)) * 1000 if times else 0.0,
        "memory_usage_mb": memory_after - memory_before,
        "bleu4_mean": bleu_mean,
        "meteor_mean": meteor_mean,
        "samples_evaluated": len(generated_all),
    }


def run(models: dict, dataloader, tokenizer, device, max_batches: int = 50) -> dict:
    """Compare every model in ``models`` on the same batches."""
    print("\nExperiment 1: backbone comparison")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        results[name] = measure_model_performance(
            model, dataloader, tokenizer, name, device, max_batches=max_batches
        )

    for name, r in results.items():
        print(f"\n{name}")
        print(f"  inference : {r['avg_inference_time_ms']:.2f} ms/batch")
        print(f"  memory    : {r['memory_usage_mb']:.1f} MB")
        print(f"  BLEU-4    : {r['bleu4_mean']:.4f}")
        print(f"  METEOR    : {r['meteor_mean']:.4f}")

    return results
