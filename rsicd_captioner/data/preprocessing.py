"""Image preprocessing for overhead imagery."""

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Resize to the backbone's input size and apply ImageNet normalisation.

    Why ImageNet statistics for satellite images?

    Both encoders start from ImageNet-pretrained weights, and those weights
    expect inputs whose channels have already been shifted and scaled by the
    ImageNet mean/std. Feeding raw [0, 1] pixels instead would put every
    activation off-distribution and throw away most of the benefit of
    pretraining. Recomputing RSICD-specific statistics is a reasonable
    alternative, but it only pays off once the backbone is fine-tuned enough to
    forget its original input scale.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Return a normalised ``H x W x C`` float array.

        LANCZOS resampling is used because RSICD images arrive at mixed
        resolutions and downsampling with a cheaper filter visibly smears the
        fine structures (road markings, roof edges) the captions describe.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return (array - self.mean) / self.std

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        """Undo normalisation so an array can be shown with ``imshow``."""
        return np.clip(array * self.std + self.mean, 0.0, 1.0)
