"""Training and validation loops with mixed precision and gradient accumulation."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .memory import cleanup, gpu_memory


def unpack_batch(batch):
    """Accept either ``(images, captions)`` or ``(images, captions, lengths)``.

    The dataset returns caption lengths for the analysis code, but the training
    loop does not need them; tolerating both shapes keeps one loop usable for
    every experiment in the project.
    """
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
    log_interval: int | None = None,
) -> float:
    """Run one training epoch and return the mean loss.

    Two things are worth knowing about this loop:

    * **Gradient accumulation.** The loss is divided by ``accumulation_steps``
      before ``backward()`` so that summing several micro-batch gradients gives
      the same result as one large batch, rather than a gradient that is N
      times too big.
    * **Mixed precision.** ``GradScaler`` multiplies the loss up before the
      backward pass so small fp16 gradients do not flush to zero, then unscales
      before the optimiser step.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if log_interval is None:
        log_interval = max(1, num_batches // 10)

    optimizer.zero_grad()
    accumulated = 0

    for batch_idx, batch in enumerate(dataloader):
        images, captions, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        try:
            with autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(images, captions)
                # The model predicts positions 1..n from inputs 0..n-1, so the
                # targets are the captions shifted left by one (<bos> dropped).
                targets = captions[:, 1:]
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated += 1
            if accumulated >= accumulation_steps:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            total_loss += loss.item() * accumulation_steps

        except RuntimeError as exc:
            # Usually an OOM on an unlucky batch. Dropping it and clearing the
            # allocator is far better than losing the whole epoch.
            print(f"  batch {batch_idx} failed ({str(exc)[:60]}...) - skipping")
            optimizer.zero_grad()
            accumulated = 0
            cleanup()
            continue

        if batch_idx % log_interval == 0:
            allocated, _, total = gpu_memory()
            progress = (batch_idx + 1) / num_batches * 100
            gpu_note = f" | GPU {allocated:.1f}/{total:.1f} GB" if total else ""
            print(f"  batch {batch_idx + 1:4d}/{num_batches} ({progress:5.1f}%) "
                  f"| loss {loss.item() * accumulation_steps:.4f}{gpu_note}")

    # Any gradients left over from a partial accumulation group would otherwise
    # be silently discarded at the start of the next epoch.
    if accumulated > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Return mean validation loss (teacher forcing, same as training)."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        images, captions, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    learning_rate: float,
    training_config,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Train one model end to end and return its loss history.

    The model is moved to the GPU on entry and back to CPU on exit so that a
    caller can train all four variants in sequence without ever holding more
    than one on the device.
    """
    print(f"\nTraining {model_name}")
    print("-" * 50)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.scheduler_step_size,
        gamma=training_config.scheduler_gamma,
    )
    use_amp = training_config.mixed_precision and device.type == "cuda"
    scaler = GradScaler(device.type) if use_amp else None

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, training_config.epochs + 1):
        print(f"\nEpoch {epoch}/{training_config.epochs} | lr {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, accumulation_steps=training_config.accumulation_steps,
        )
        val_loss = validate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        cleanup()

    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{model_name.replace('+', '_').lower()}.pt"
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name}, path)
        print(f"  checkpoint saved to {path}")

    # Hand the GPU back before the next model is trained.
    model.to("cpu")
    cleanup()

    print(f"  best val loss {best_val:.4f} (epoch {best_epoch})")

    return {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }
