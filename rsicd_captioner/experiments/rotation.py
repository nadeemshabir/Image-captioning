"""Experiment 2 - rotation invariance.

Overhead imagery has no canonical "up": the same parking lot photographed on a
different orbit pass is the same scene rotated. A captioner that describes it
differently at 0 deg and 90 deg has latched onto orientation rather than
content, so consistency across rotations is a genuine quality signal here in a
way it would not be for ground-level photography.
"""

import ast
import io
import random
import re

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..training.memory import cleanup

# Keywords used to spread the sample across land-use types. RSICD is dominated
# by a handful of scene classes, and sampling naively returns airport after
# airport - which tells you nothing about how the model handles anything else.
SCENE_KEYWORDS = [
    "residential", "house", "building", "road", "forest", "trees", "water",
    "river", "lake", "beach", "coast", "mountain", "hill", "field",
    "farm", "agriculture", "crop", "desert", "industrial", "factory",
    "park", "green", "urban", "rural", "harbor", "port", "bridge",
    "school", "church", "stadium", "parking", "railway", "train",
]

AIRPORT_KEYWORDS = ["airport", "runway", "terminal", "plane", "aircraft"]


def get_diverse_scene_indices(dataset, max_images: int = 6, seed: int | None = None) -> list[int]:
    """Pick dataset indices covering as many distinct scene types as possible."""
    if seed is not None:
        random.seed(seed)

    by_keyword: dict[str, list[int]] = {k: [] for k in SCENE_KEYWORDS}
    airports: list[int] = []
    others: list[int] = []
    seen_rows: set[int] = set()

    for ds_idx, row_idx in enumerate(dataset.indices):
        # One entry per image, not per caption.
        if row_idx in seen_rows:
            continue
        seen_rows.add(row_idx)

        try:
            row = dataset.df.iloc[row_idx]
            text = " ".join(re.findall(r"'([^']*)'", str(row["captions"]))).lower()
        except Exception:
            others.append(ds_idx)
            continue

        is_airport = any(w in text for w in AIRPORT_KEYWORDS)
        matched = next((k for k in SCENE_KEYWORDS if k in text and not is_airport), None)

        if matched:
            by_keyword[matched].append(ds_idx)
        elif is_airport:
            airports.append(ds_idx)
        else:
            others.append(ds_idx)

    # One image per scene type first, then fill from the unclassified pool,
    # and only fall back to airports if there is still room.
    selected = [random.choice(v) for v in by_keyword.values() if v][:max_images]
    for pool in (others, airports):
        random.shuffle(pool)
        for idx in pool:
            if len(selected) >= max_images:
                break
            selected.append(idx)

    random.shuffle(selected)
    return selected[:max_images]


def fetch_original_image(dataset, ds_idx: int):
    """Recover the source PIL image, filename and reference captions for an item."""
    row_idx = dataset.indices[ds_idx]
    row = dataset.df.iloc[row_idx]
    image_dict = ast.literal_eval(row["image"])
    image = Image.open(io.BytesIO(image_dict["bytes"])).convert("RGB")
    captions = [c.strip() for c in re.findall(r"'([^']*)'", str(row["captions"])) if c.strip()]
    return image, row.get("filename", f"row_{row_idx}"), captions


@torch.no_grad()
def evaluate_rotation_invariance(model, dataset, tokenizer, model_name, device,
                                 num_images: int = 6, seed: int = 42) -> dict:
    """Caption each image at 0/90/180/270 deg and measure caption agreement.

    Agreement is the mean Jaccard overlap of the word sets against the 0 deg
    caption. Word-set overlap rather than exact match, because a robust model
    is allowed to reorder a phrase - it just should not start describing a
    different scene.
    """
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.float()

    indices = get_diverse_scene_indices(dataset, max_images=num_images, seed=seed)
    per_angle: dict[int, list[str]] = {0: [], 90: [], 180: [], 270: []}
    samples = []

    for ds_idx in indices:
        image_tensor, _, _ = dataset[ds_idx]
        image_tensor = image_tensor.float().unsqueeze(0).to(device)
        pil_image, filename, references = fetch_original_image(dataset, ds_idx)

        rotation_captions: dict[int, str] = {}
        rotation_images: dict[int, Image.Image] = {}

        for angle in (0, 90, 180, 270):
            if angle == 0:
                rotated, rotated_pil = image_tensor, pil_image
            else:
                rotated = TF.rotate(image_tensor, angle)
                # PIL rotates counter-clockwise, torchvision clockwise; the
                # negation keeps the displayed image matching the tensor.
                rotated_pil = pil_image.rotate(-angle, expand=True)

            generated = model.generate_caption(rotated, tokenizer, max_length=20)
            caption = generated[0] if generated else ""
            rotation_captions[angle] = caption
            rotation_images[angle] = rotated_pil
            per_angle[angle].append(caption)

        samples.append({
            "dataset_idx": ds_idx,
            "filename": filename,
            "references": references,
            "rotation_captions": rotation_captions,
            "rotation_images": rotation_images,
        })

    consistency_scores = []
    for i in range(len(per_angle[0])):
        base = set(per_angle[0][i].split())
        overlaps = []
        for angle in (90, 180, 270):
            other = set(per_angle[angle][i].split())
            if base and other:
                overlaps.append(len(base & other) / len(base | other))
        if overlaps:
            consistency_scores.append(float(np.mean(overlaps)))

    model.to("cpu")
    cleanup()

    return {
        "model_name": model_name,
        "consistency_score": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
        "samples": samples,
    }


def visualize(result: dict, max_samples: int = 4) -> None:
    """Plot each sample's four rotations with the caption produced for each."""
    import matplotlib.pyplot as plt

    samples = result["samples"][:max_samples]
    if not samples:
        print("nothing to visualise")
        return

    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, 4)

    for r, sample in enumerate(samples):
        for c, angle in enumerate((0, 90, 180, 270)):
            ax = axes[r, c]
            ax.imshow(sample["rotation_images"][angle])
            ax.set_title(f"{angle} deg: {sample['rotation_captions'][angle][:40]}", fontsize=8)
            ax.axis("off")

    plt.suptitle(f"Rotation invariance - {result['model_name']} "
                 f"(consistency {result['consistency_score']:.3f})")
    plt.tight_layout()
    plt.show()


def run(models: dict, dataset, tokenizer, device, num_images: int = 6) -> dict:
    """Run the rotation test over a set of models."""
    print("\nExperiment 2: rotation invariance")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_rotation_invariance(
            model, dataset, tokenizer, name, device, num_images=num_images
        )
        print(f"{name}: consistency {results[name]['consistency_score']:.4f}")
    return results
