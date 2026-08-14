"""Experiment 5 - how many memory tokens should the image become?

A Transformer decoder cross-attends to an encoder-side sequence. With a single
image vector that "sequence" has length 1, so attention has nothing to choose
between and effectively collapses to a bias term. Expanding the image into
several learned tokens gives attention something to select from - the question
is whether the decoder actually uses that freedom or just overfits faster.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..training.loops import unpack_batch
from ..training.memory import cleanup
from .vision_text_interface import build_feature_extractor


class TransformerDecoderNToken(nn.Module):
    """Transformer decoder whose image memory is ``memory_tokens`` long."""

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 feature_dim: int, memory_tokens: int = 1, max_positions: int = 100):
        super().__init__()
        self.d_model = d_model
        self.memory_tokens = memory_tokens

        self.embedding = nn.Embedding(vocab_size, d_model)
        # Learned positions here rather than sinusoids, to keep this variant
        # minimal and isolate the memory-size effect.
        self.pos_encoding = nn.Embedding(max_positions, d_model)
        self.memory_projection = nn.Linear(feature_dim, memory_tokens * d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, features, captions):
        batch_size = captions.size(0)
        seq_len = captions.size(1) - 1  # last token has no target to predict

        memory = self.memory_projection(features).view(
            batch_size, self.memory_tokens, self.d_model
        )

        tgt = self.embedding(captions[:, :-1])
        positions = torch.arange(seq_len, device=captions.device).unsqueeze(0).expand(batch_size, -1)
        tgt = tgt + self.pos_encoding(positions)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(captions.device)
        output = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        return self.output_projection(output)


def create_memory_variants(vocab_size: int, d_model: int = 256, nhead: int = 4,
                           num_layers: int = 2) -> dict:
    """Build the 1-token and 4-token variants over a shared encoder design."""
    extractor, feature_dim = build_feature_extractor()

    return {
        "Transformer_1Token": nn.Sequential(
            extractor,
            TransformerDecoderNToken(vocab_size, d_model, nhead, num_layers, feature_dim, 1),
        ),
        "Transformer_4Token": nn.Sequential(
            extractor,
            TransformerDecoderNToken(vocab_size, d_model, nhead, num_layers, feature_dim, 4),
        ),
    }


def compare(models_dict: dict, train_dataset, val_dataset, tokenizer, device,
            epochs: int = 3, train_samples: int = 300, val_samples: int = 100,
            batch_size: int = 8, lr: float = 5e-4) -> dict:
    """Train both variants on the same small slice and compare train/val loss.

    Watching the *gap* between train and val matters more than either number:
    extra memory capacity that only improves training loss is capacity being
    spent on memorisation.
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
                outputs = model[1](model[0](images), captions)
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
                    outputs = model[1](model[0](images), captions)
                    val_running += criterion(
                        outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1)
                    ).item()

            train_losses.append(running / max(1, len(train_loader)))
            val_losses.append(val_running / max(1, len(val_loader)))
            print(f"  epoch {epoch + 1}: train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}")

        results[name] = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "convergence": train_losses[0] - train_losses[-1],
        }
        model.to("cpu")
        cleanup()

    return results


def run(train_dataset, val_dataset, tokenizer, device, epochs: int = 3) -> dict:
    """Experiment 5 entry point."""
    print("\nExperiment 5: transformer memory tokens (1 vs 4)")
    print("=" * 60)

    variants = create_memory_variants(len(tokenizer))
    results = compare(variants, train_dataset, val_dataset, tokenizer, device, epochs)

    for name, r in results.items():
        print(f"{name}: train {r['final_train_loss']:.4f}, val {r['final_val_loss']:.4f}")
    return results
