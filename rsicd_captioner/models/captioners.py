"""End-to-end captioning models (encoder + decoder) and a factory."""

import torch
import torch.nn as nn

from .decoders import LSTMDecoder, TransformerDecoder
from .encoder import CNNEncoder


class _BaseCaptioner(nn.Module):
    """Shared plumbing: run the encoder, hand off to the decoder, decode text."""

    encoder: CNNEncoder
    decoder: nn.Module

    def forward(self, images, captions=None, max_length: int = 24):
        image_features = self.encoder(images)
        return self.decoder(image_features, captions, max_length)

    @torch.no_grad()
    def generate_caption(self, images, tokenizer, max_length: int = 24) -> list[str]:
        """Greedy-decode one caption per image and return them as strings."""
        self.eval()
        logits = self.forward(images, captions=None, max_length=max_length)
        predictions = logits.argmax(dim=-1)
        return [tokenizer.decode_caption(pred.tolist()) for pred in predictions]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CNNLSTMCaptioner(_BaseCaptioner):
    """CNN encoder + LSTM decoder."""

    def __init__(
        self,
        vocab_size: int,
        encoder_name: str = "resnet18",
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = CNNEncoder(encoder_name, embed_dim, pretrained)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)


class CNNTransformerCaptioner(_BaseCaptioner):
    """CNN encoder + Transformer decoder."""

    def __init__(
        self,
        vocab_size: int,
        encoder_name: str = "resnet18",
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        memory_tokens: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = CNNEncoder(encoder_name, embed_dim, pretrained)
        self.decoder = TransformerDecoder(
            vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, memory_tokens
        )


def build_model(vocab_size: int, encoder_name: str, decoder_type: str, model_config):
    """Construct one of the four encoder/decoder pairings from a config."""
    if decoder_type == "lstm":
        return CNNLSTMCaptioner(
            vocab_size,
            encoder_name=encoder_name,
            embed_dim=model_config.embed_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            dropout=model_config.lstm_dropout,
            pretrained=model_config.pretrained,
        )
    if decoder_type == "transformer":
        return CNNTransformerCaptioner(
            vocab_size,
            encoder_name=encoder_name,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_decoder_layers,
            ff_dim=model_config.ff_dim,
            dropout=model_config.transformer_dropout,
            memory_tokens=model_config.memory_tokens,
            pretrained=model_config.pretrained,
        )
    raise ValueError(f"Unknown decoder type: {decoder_type!r} (expected 'lstm' or 'transformer')")


def build_all_variants(vocab_size: int, model_config) -> dict[str, nn.Module]:
    """Build all four variants on CPU.

    They are deliberately left on CPU: holding four captioners on the GPU at
    once wastes VRAM that the training batch needs, so the training script
    moves them across one at a time.
    """
    from ..config import MODEL_VARIANTS

    return {
        name: build_model(vocab_size, encoder, decoder, model_config)
        for name, (encoder, decoder) in MODEL_VARIANTS.items()
    }
