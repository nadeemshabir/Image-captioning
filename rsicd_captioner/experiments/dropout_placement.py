"""Experiment 6 - where should dropout go in the decoder?

Two placements with the same rate regularise different things:

* **On the embeddings** - randomly blanks input words, forcing the decoder to
  rely on the image and on longer context rather than on the immediately
  preceding word.
* **Inside the recurrent block** - perturbs the hidden representation itself,
  regularising how the state is composed rather than what goes into it.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..training.loops import unpack_batch
from ..training.memory import cleanup
from .vision_text_interface import build_feature_extractor


class LSTMWithEmbeddingDropout(nn.Module):
    """Dropout applied to the word embeddings before they reach the LSTM."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        h0 = self.feature_projection(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        embeds = self.embed_dropout(self.embedding(captions[:, :-1]))
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        return self.output_projection(lstm_out)


class LSTMWithDecoderDropout(nn.Module):
    """Dropout applied on the LSTM output, i.e. inside the decoder block.

    Note that ``nn.LSTM(dropout=...)`` only drops *between stacked layers* and
    does nothing for a single-layer LSTM, so the dropout is applied explicitly
    to the output instead - otherwise this variant would be an unregularised
    control by accident.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder_dropout = nn.Dropout(dropout)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        h0 = self.feature_projection(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        embeds = self.embedding(captions[:, :-1])
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        return self.output_projection(self.decoder_dropout(lstm_out))


def create_dropout_variants(vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512,
                            dropout: float = 0.3) -> dict:
    extractor, feature_dim = build_feature_extractor()
    return {
        "LSTM_EmbedDropout": nn.Sequential(
            extractor,
            LSTMWithEmbeddingDropout(vocab_size, embed_dim, hidden_dim, feature_dim, dropout),
        ),
        "LSTM_DecoderDropout": nn.Sequential(
            extractor,
            LSTMWithDecoderDropout(vocab_size, embed_dim, hidden_dim, feature_dim, dropout),
        ),
    }


def compare(models_dict: dict, train_dataset, val_dataset, tokenizer, device,
            epochs: int = 3, train_samples: int = 400, val_samples: int = 100,
            batch_size: int = 16, lr: float = 1e-3) -> dict:
    """Train both placements and report the train/val gap for each."""
    n_train = min(train_samples, len(train_dataset))
    n_val = min(val_samples, len(val_dataset))
    train_loader = DataLoader(Subset(train_dataset, range(0, n_train)),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, range(0, n_val)),
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
        for _ in range(epochs):
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

        results[name] = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            # Negative gap = val loss above train loss, i.e. the usual
            # generalisation shortfall. Larger magnitude means more overfitting.
            "overfitting_gap": train_losses[-1] - val_losses[-1],
            "regularization_effect": val_losses[0] - val_losses[-1],
        }
        print(f"  train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}, "
              f"gap {results[name]['overfitting_gap']:.4f}")

        model.to("cpu")
        cleanup()

    return results


def run(train_dataset, val_dataset, tokenizer, device, epochs: int = 3) -> dict:
    """Experiment 6 entry point."""
    print("\nExperiment 6: dropout placement")
    print("=" * 60)

    variants = create_dropout_variants(len(tokenizer))
    return compare(variants, train_dataset, val_dataset, tokenizer, device, epochs)
