"""CNN image encoder."""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class CNNEncoder(nn.Module):
    """Turn a 224x224 image into a single ``embed_dim`` feature vector.

    Two backbones are supported so that the accuracy/cost trade-off can be
    measured directly:

    * **ResNet-18** - 512-d pooled features, ~11.4M parameters. The middle
      ground: enough capacity to separate visually similar land-use classes
      without the memory cost of a deeper ResNet.
    * **MobileNet-v2** - 1280-d pooled features, ~2.9M parameters. Depthwise
      separable convolutions make it roughly 4x smaller.

    Both are projected down to a shared ``embed_dim`` so either encoder can be
    dropped into either decoder without touching the decoder's code.
    """

    def __init__(self, model_name: str = "resnet18", embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name

        if model_name == "resnet18":
            self.cnn = tv_models.resnet18(pretrained=pretrained)
            # Replacing the classifier with Identity leaves the global average
            # pool in place, so forward() returns pooled features directly.
            self.cnn.fc = nn.Identity()
            self.feature_dim = 512
        elif model_name == "mobilenet":
            self.cnn = tv_models.mobilenet_v2(pretrained=pretrained)
            self.cnn.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported encoder: {model_name!r} (expected 'resnet18' or 'mobilenet')")

        self.feature_proj = nn.Linear(self.feature_dim, embed_dim)
        # Light dropout on the image vector: it is the single point through
        # which all visual information reaches the decoder, so the decoder will
        # happily overfit to it if left unregularised.
        self.dropout = nn.Dropout(0.1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` -> ``[B, embed_dim]``."""
        features = self.cnn(images)
        return self.dropout(self.feature_proj(features))
