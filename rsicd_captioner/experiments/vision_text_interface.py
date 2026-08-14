"""Experiments 3 & 4 - how image features should enter the decoder.

Experiment 3 compares the two trained decoder families on identical batches.
Experiment 4 isolates the interface question inside the LSTM family: should the
image initialise the recurrent state, or be fed in as an extra first token?
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Subset

from ..evaluation.metrics import aggregate, compute_bleu4, tokens_from_indices
from ..training.loops import unpack_batch
from ..training.memory import cleanup


# --------------------------------------------------------------------------- #
# Experiment 3: compare the trained decoder families
# --------------------------------------------------------------------------- #

@torch.no_grad()
def quick_eval(model, dataset, tokenizer, device, num_samples: int = 50,
               batch_size: int = 8) -> dict:
    """BLEU-4 over a small fixed slice - cheap enough to run on every variant."""
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.float()

    subset = Subset(dataset, range(0, min(num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    generated_all, references_all = [], []
    for batch in loader:
        images, captions, _ = unpack_batch(batch)
        images = images.to(device, dtype=torch.float32)
        try:
            generated = model.generate_caption(images, tokenizer, max_length=20)
        except RuntimeError:
            continue
        for j, caption in enumerate(generated):
            generated_all.append(caption)
            references_all.append([tokens_from_indices(captions[j], tokenizer, keep_unk=True)])

    model.to("cpu")
    cleanup()

    scores = [compute_bleu4(g, r) for g, r in zip(generated_all, references_all)]
    mean, _ = aggregate(scores)
    return {
        "bleu4_mean": mean,
        "samples": len(generated_all),
        "sample_captions": generated_all[:3],
    }


def run_interface_comparison(models: dict, dataset, tokenizer, device,
                             num_samples: int = 50) -> dict:
    """Score every trained model on the same slice, grouped by decoder family."""
    print("\nExperiment 3: vision-text interface comparison")
    print("=" * 60)

    results = {"lstm_models": {}, "transformer_models": {}}
    for name, model in models.items():
        group = "lstm_models" if "LSTM" in name else "transformer_models"
        results[group][name] = quick_eval(model, dataset, tokenizer, device, num_samples)
        print(f"{name}: BLEU-4 {results[group][name]['bleu4_mean']:.4f}")
    return results


# --------------------------------------------------------------------------- #
# Experiment 4: init-hidden vs image-as-token, trained from scratch
# --------------------------------------------------------------------------- #

class LSTMDecoderInitHidden(nn.Module):
    """Image features become the LSTM's initial hidden state.

    The image conditions the recurrence itself, so its influence persists
    through every timestep via the carried state.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        h0 = self.feature_projection(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        embeds = self.embedding(captions[:, :-1])
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        return self.output_projection(lstm_out)


class LSTMDecoderImgToken(nn.Module):
    """Image features become a pseudo-word prepended to the caption.

    The LSTM starts from a zero state and "reads" the image as its first input.
    That costs one timestep and means the image's influence has to survive
    being carried through the recurrence rather than being injected into it.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Projects into embedding space, not hidden space - the image has to
        # look like a word here, not like a state.
        self.feature_projection = nn.Linear(feature_dim, embed_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        img_embed = self.feature_projection(features).unsqueeze(1)
        caption_embeds = self.embedding(captions[:, :-1])
        embeds = torch.cat([img_embed, caption_embeds], dim=1)

        lstm_out, _ = self.lstm(embeds)
        # Drop the output at the image position: it predicts a word from the
        # image alone, which has no target to compare against.
        return self.output_projection(lstm_out[:, 1:])


def build_feature_extractor(feature_dim: int = 512):
    """A frozen-architecture ResNet-18 trunk that outputs pooled features."""
    resnet = tv_models.resnet18(pretrained=True)
    return nn.Sequential(
        *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten()
    ), feature_dim


def create_interface_variants(vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
    """Build the two interface variants over a shared encoder design."""
    extractor, feature_dim = build_feature_extractor()
    return {
        "LSTM_InitHidden": nn.Sequential(
            extractor, LSTMDecoderInitHidden(vocab_size, embed_dim, hidden_dim, feature_dim)
        ),
        "LSTM_ImgToken": nn.Sequential(
            extractor, LSTMDecoderImgToken(vocab_size, embed_dim, hidden_dim, feature_dim)
        ),
    }


def train_and_compare(models_dict: dict, train_dataset, val_dataset, tokenizer, device,
                      epochs: int = 3, train_samples: int = 500, val_samples: int = 100,
                      batch_size: int = 16, lr: float = 1e-3) -> dict:
    """Train each variant briefly on a small slice and compare convergence.

    This is deliberately a short run on a subset. The question is which
    interface *learns faster from the same budget*, not which wins after full
    convergence, so a small controlled comparison answers it more cheaply.
    """
    train_loader = DataLoader(Subset(train_dataset, range(0, train_samples)),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, range(0, val_samples)),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_idx["<pad>"])

    results = {}
    for name, model in models_dict.items():
        print(f"Training {name}...")
        model = model.to(device)
        if device.type == "cuda":
            model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            running = 0.0
            for batch in train_loader:
                images, captions, _ = unpack_batch(batch)
                images = images.to(device, dtype=torch.float32)
                captions = captions.to(device)

                optimizer.zero_grad()
                features = model[0](images)
                outputs = model[1](features, captions)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1)
                )
                loss.backward()
                optimizer.step()
                running += loss.item()

            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images, captions, _ = unpack_batch(batch)
                    images = images.to(device, dtype=torch.float32)
                    captions = captions.to(device)
                    features = model[0](images)
                    outputs = model[1](features, captions)
                    val_running += criterion(
                        outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1)
                    ).item()

            train_losses.append(running / max(1, len(train_loader)))
            val_losses.append(val_running / max(1, len(val_loader)))
            print(f"  epoch {epoch + 1}: train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}")

        results[name] = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "convergence_speed": train_losses[0] - train_losses[-1],
        }
        model.to("cpu")
        cleanup()

    return results


def run_lstm_interface(train_dataset, val_dataset, tokenizer, device, epochs: int = 3) -> dict:
    """Experiment 4 entry point."""
    print("\nExperiment 4: LSTM vision-text interface (init-hidden vs image token)")
    print("=" * 60)

    variants = create_interface_variants(len(tokenizer))
    results = train_and_compare(variants, train_dataset, val_dataset, tokenizer, device, epochs)

    for name, r in results.items():
        print(f"{name}: train {r['final_train_loss']:.4f}, val {r['final_val_loss']:.4f}, "
              f"convergence {r['convergence_speed']:.4f}")
    return results
