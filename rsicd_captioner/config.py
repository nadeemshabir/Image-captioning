"""Central configuration for the RSICD captioning pipeline.

Everything that a person might reasonably want to tune lives here, so that no
magic number is buried three call-frames deep inside a training loop.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    """Where the data lives and how it is turned into tensors.

    The RSICD CSVs store each image as a stringified dict holding raw JPEG
    bytes (``{'bytes': b'...'}``), which is why the dataset parses them with
    ``ast.literal_eval`` rather than reading files off disk.
    """

    root: Path = Path("data")
    train_csv: str = "train.csv"
    valid_csv: str = "valid.csv"
    test_csv: str = "test.csv"

    # Images are resized to the resolution the ImageNet backbones were trained
    # on. Anything else would waste the pretrained weights.
    image_size: tuple[int, int] = (224, 224)

    # ImageNet channel statistics. We keep them (rather than computing RSICD's
    # own mean/std) because both backbones are initialised from ImageNet
    # weights, and matching the normalisation is what makes transfer learning
    # work in the first place.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Every RSICD image ships with 5 human-written captions. Using all 5 gives
    # the decoder 5x more supervision from the same set of images; lowering
    # this is a cheap way to study how much caption diversity actually helps.
    max_captions_per_image: int = 5

    @property
    def train_path(self) -> Path:
        return self.root / self.train_csv

    @property
    def valid_path(self) -> Path:
        return self.root / self.valid_csv

    @property
    def test_path(self) -> Path:
        return self.root / self.test_csv


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

@dataclass
class TokenizerConfig:
    """Word-level vocabulary settings.

    ``vocab_size`` is an upper bound, not a target: RSICD's caption language is
    narrow enough that a ``min_word_freq`` of 2 leaves roughly 1.8k distinct
    words, well under the cap.
    """

    vocab_size: int = 10_000
    max_length: int = 24
    min_word_freq: int = 2

    # Reserved indices, fixed by construction order in CaptionTokenizer.
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    """Architecture hyper-parameters shared by both decoder families."""

    encoder_name: str = "resnet18"      # "resnet18" | "mobilenet"
    embed_dim: int = 512
    pretrained: bool = True

    # LSTM decoder
    hidden_dim: int = 512
    num_layers: int = 1
    lstm_dropout: float = 0.2

    # Transformer decoder
    num_heads: int = 8
    num_decoder_layers: int = 4
    ff_dim: int = 2048
    transformer_dropout: float = 0.1
    # How many "memory" vectors the image is expanded into for cross-attention.
    # 1 forces the whole scene through a single vector; more tokens give the
    # decoder somewhere to spread different aspects of the image.
    memory_tokens: int = 2


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainingConfig:
    """Training loop settings, tuned for a single 16 GB GPU."""

    epochs: int = 5
    batch_size: int = 256
    # Effective batch = batch_size * accumulation_steps. Accumulating lets us
    # get large-batch behaviour without the activation memory of a large batch.
    accumulation_steps: int = 2
    num_workers: int = 4

    # Transformers are more sensitive to a high LR at this batch size, so they
    # get half the learning rate the LSTMs do.
    lstm_lr: float = 1e-4
    transformer_lr: float = 5e-5

    scheduler_step_size: int = 3
    scheduler_gamma: float = 0.7

    # FP16 autocast + GradScaler. Roughly halves activation memory and speeds
    # up the convolution-heavy encoder considerably.
    mixed_precision: bool = True

    # Free cached GPU blocks every N batches so long runs do not creep upward.
    cleanup_interval: int = 50

    checkpoint_dir: Path = Path("checkpoints")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvalConfig:
    """Caption generation and scoring settings."""

    max_gen_length: int = 25
    eval_batch_size: int = 8
    # Cap on how many validation samples the reported metrics are computed over.
    eval_samples: int = 800
    # Length-penalty sweep used by the decoding experiment.
    length_penalties: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)


# --------------------------------------------------------------------------- #
# Top-level bundle
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42


# The four encoder/decoder pairings this project compares.
MODEL_VARIANTS: dict[str, tuple[str, str]] = {
    "ResNet+LSTM": ("resnet18", "lstm"),
    "MobileNet+LSTM": ("mobilenet", "lstm"),
    "ResNet+Transformer": ("resnet18", "transformer"),
    "MobileNet+Transformer": ("mobilenet", "transformer"),
}


def get_device() -> torch.device:
    """Pick the best available accelerator and switch on cuDNN autotuning.

    Autotuning pays off here because every batch has the same 224x224 shape,
    so cuDNN's chosen algorithm stays valid for the whole run.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
