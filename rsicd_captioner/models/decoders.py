"""Caption decoders: an LSTM baseline and a Transformer alternative."""

import math

import torch
import torch.nn as nn

# Special-token indices, fixed by CaptionTokenizer's construction order.
PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2


class LSTMDecoder(nn.Module):
    """Autoregressive LSTM decoder conditioned on the image feature vector.

    **How the image enters the decoder.** The image vector is projected into
    the LSTM's initial hidden *and* cell state, rather than being prepended as
    a pseudo-word. Both work; conditioning the state is the cheaper of the two
    (no extra timestep, no vocabulary slot) and keeps the image visible to the
    recurrence at every step instead of only at step zero. The alternative is
    implemented in ``experiments/vision_text_interface.py`` so the two can be
    compared head to head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Separate projections for h0 and c0: the hidden state carries "what to
        # say next" while the cell state carries longer-range content, and
        # forcing them to share a projection measurably slows early training.
        self.image_to_hidden = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.image_to_cell = nn.Linear(embed_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            # nn.LSTM only applies dropout *between* layers, so it is a no-op
            # (and emits a warning) for a single-layer stack.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for the embedding and output layers, zeros for <pad>."""
        nn.init.xavier_uniform_(self.word_embed.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        with torch.no_grad():
            # A non-zero <pad> embedding would let padding leak signal into the
            # recurrence on short captions.
            self.word_embed.weight[self.pad_idx].fill_(0)

    def forward(self, image_features, captions=None, max_length: int = 24):
        """Teacher forcing when ``captions`` is given, greedy decoding otherwise."""
        batch_size = image_features.size(0)

        # [B, hidden*layers] -> [layers, B, hidden], the layout nn.LSTM expects.
        h0 = (self.image_to_hidden(image_features)
              .view(batch_size, self.num_layers, self.hidden_dim)
              .transpose(0, 1).contiguous())
        c0 = (self.image_to_cell(image_features)
              .view(batch_size, self.num_layers, self.hidden_dim)
              .transpose(0, 1).contiguous())

        if captions is not None:
            # Training: feed the ground-truth prefix and predict the next word
            # at every position. The final token is dropped from the input
            # because there is nothing left to predict after it.
            inputs = captions[:, :-1]
            embeds = self.dropout(self.word_embed(inputs))
            lstm_out, _ = self.lstm(embeds, (h0, c0))
            return self.output_proj(self.dropout(lstm_out))

        # Inference: step the LSTM one word at a time, feeding back its own
        # argmax. Hidden state carries across steps so the prefix is not
        # re-encoded each time.
        outputs = []
        hidden = (h0, c0)
        input_word = torch.full(
            (batch_size, 1), BOS_IDX, dtype=torch.long, device=image_features.device
        )

        for _ in range(max_length):
            embed = self.dropout(self.word_embed(input_word))
            lstm_out, hidden = self.lstm(embed, hidden)
            logits = self.output_proj(self.dropout(lstm_out))
            outputs.append(logits)

            input_word = logits.argmax(dim=-1)
            if (input_word == EOS_IDX).all():
                break  # every sequence in the batch has finished

        return torch.cat(outputs, dim=1)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal added to word embeddings.

    Self-attention is permutation-invariant, so without this the decoder would
    treat "road crosses river" and "river crosses road" as identical inputs.
    The sinusoidal form is used rather than learned positions because captions
    are short and fixed-length here, leaving nothing for a learned table to
    discover that the sinusoids do not already encode.
    """

    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer: saved with the model, moved by .to(), not a parameter.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerDecoder(nn.Module):
    """Transformer decoder that cross-attends to the image as a short memory.

    The image vector is expanded into ``memory_tokens`` vectors which act as
    the encoder-side sequence for cross-attention. With a single token the
    decoder must squeeze the whole scene through one vector; with several, it
    can attend to different learned aspects at different points in the
    sentence. ``experiments/memory_tokens.py`` measures whether that extra
    room actually helps.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        memory_tokens: int = 2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_tokens = memory_tokens
        self.pad_idx = pad_idx

        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim)

        self.image_to_memory = nn.Linear(embed_dim, embed_dim * memory_tokens)
        # LayerNorm on the memory keeps its scale comparable to the word
        # embeddings; without it, attention logits are dominated by whichever
        # side happens to have the larger magnitude early in training.
        self.memory_layer_norm = nn.LayerNorm(embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, num_heads, ff_dim, dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        """Upper-triangular -inf mask so position i cannot attend to i+1 onwards."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, image_features, captions=None, max_length: int = 24):
        batch_size = image_features.size(0)
        device = image_features.device

        memory = self.image_to_memory(image_features)
        memory = self.memory_layer_norm(
            memory.view(batch_size, self.memory_tokens, self.embed_dim)
        )

        if captions is not None:
            inputs = captions[:, :-1]
            embeds = self.dropout(self.pos_encoding(self.word_embed(inputs)))

            # Two masks with different jobs: the causal mask prevents looking
            # ahead, the padding mask stops attention from spending weight on
            # <pad> positions in shorter captions.
            tgt_mask = self._causal_mask(inputs.size(1), device)
            tgt_key_padding_mask = inputs == self.pad_idx

            decoded = self.transformer(
                embeds, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
            )
            return self.output_proj(decoded)

        # Inference: unlike the LSTM there is no carried state, so the growing
        # prefix is re-encoded each step and only the last position is kept.
        outputs = []
        generated = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_length):
            embeds = self.dropout(self.pos_encoding(self.word_embed(generated)))
            tgt_mask = self._causal_mask(generated.size(1), device)

            decoded = self.transformer(embeds, memory, tgt_mask=tgt_mask)
            next_logits = self.output_proj(decoded[:, -1:])
            outputs.append(next_logits)

            next_word = next_logits.argmax(dim=-1)
            generated = torch.cat([generated, next_word], dim=1)
            if (next_word == EOS_IDX).all():
                break

        return torch.cat(outputs, dim=1)
