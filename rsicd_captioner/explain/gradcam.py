"""Grad-CAM for captioning models.

Grad-CAM answers "which pixels moved this prediction?" by weighting the last
convolutional feature maps by the gradient flowing into them. Applying it to a
captioner needs one adaptation: there is no single class score to differentiate,
so we take the logit of a chosen output word and trace its gradient back through
the decoder into the encoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Hook a CNN layer and produce a class-activation heatmap."""

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks: list = []
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Locate the named layer, falling back to the last conv layer."""
        cnn = getattr(getattr(self.model, "encoder", self.model), "cnn", None)
        if cnn is None:
            raise ValueError("could not locate a CNN inside this model")

        named = dict(cnn.named_modules())
        if self.target_layer_name in named:
            return named[self.target_layer_name]

        # MobileNet has no "layer4"; its last conv is the right analogue.
        conv_layers = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise ValueError("no convolutional layers found")
        return conv_layers[-1]

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inp, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        layer = self._find_target_layer()
        self.hooks.append(layer.register_forward_hook(forward_hook))
        self.hooks.append(layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, image: torch.Tensor, target_word_idx: int | None = None):
        """Return a normalised heatmap for one image ``[1, 3, H, W]``.

        The model is put in ``train()`` mode: cuDNN's fused RNN kernels do not
        retain what they need for a backward pass in eval mode, so an LSTM
        decoder would otherwise raise as soon as ``backward()`` is called.
        Dropout being active adds a little noise but does not change which
        regions dominate.
        """
        self.model.train()
        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)
        image = image.clone().requires_grad_(True)

        features = self.model.encoder(image)
        # A minimal <bos> word <eos> sequence puts the decoder in its training
        # code path, which is the one gradients actually flow through.
        dummy_caption = torch.tensor([[1, 5, 2]], dtype=torch.long, device=image.device)
        outputs = self.model.decoder(features, dummy_caption)

        if target_word_idx is None:
            target_word_idx = outputs[0, 0].argmax().item()
        score = outputs[0, 0, target_word_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return torch.zeros(7, 7), target_word_idx

        # Channel importance = spatially averaged gradient; the heatmap is the
        # weighted sum of feature maps, with negatives clipped because only
        # positive evidence for the word is of interest.
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True)).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_word_idx

    def cleanup(self) -> None:
        """Remove hooks - leaving them attached leaks memory across models."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def visualize_gradcam(model, dataset, tokenizer, indices, model_name, device,
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
    """Plot each selected image beside its Grad-CAM overlay and caption."""
    import matplotlib.pyplot as plt

    model.to(device)
    if device.type == "cuda":
        model.float()
    gradcam = GradCAM(model, target_layer_name="layer4")

    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        try:
            image_tensor, _, _ = dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                generated = model.generate_caption(image_batch, tokenizer, max_length=15)
            caption = generated[0] if generated else ""

            cam, _ = gradcam.generate_cam(image_batch)

            image_np = image_tensor.float().permute(1, 2, 0).numpy()
            image_np = np.clip(image_np * np.array(std) + np.array(mean), 0, 1)

            # Upsample the coarse (typically 7x7) map to image resolution.
            cam_resized = (
                F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2],
                    mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
            )

            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f"image #{idx}", fontsize=9)
            axes[0, i].axis("off")

            axes[1, i].imshow(image_np)
            axes[1, i].imshow(cam_resized, alpha=0.4, cmap="jet")
            axes[1, i].set_title(caption[:30], fontsize=8)
            axes[1, i].axis("off")

        except Exception as exc:
            print(f"  sample {idx} failed: {str(exc)[:50]}")
            axes[0, i].axis("off")
            axes[1, i].axis("off")

    plt.suptitle(f"Grad-CAM - {model_name}")
    plt.tight_layout()
    plt.show()

    gradcam.cleanup()
    model.eval()
    model.to("cpu")
