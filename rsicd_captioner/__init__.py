"""RSICD-Captioner: image captioning for remote-sensing imagery."""

__version__ = "0.1.0"

from .config import Config, get_device, MODEL_VARIANTS

__all__ = ["Config", "get_device", "MODEL_VARIANTS", "__version__"]
