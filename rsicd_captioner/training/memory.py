"""Memory monitoring and cleanup helpers.

Long sequential training runs on a single GPU tend to die from slow leaks
rather than from one oversized batch, so these are used liberally between
models and epochs.
"""

import gc

import torch


def gpu_memory() -> tuple[float, float, float]:
    """Return ``(allocated_gb, reserved_gb, total_gb)`` for the current device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return allocated, reserved, total


def report_memory() -> None:
    """Print a system + GPU memory snapshot."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB "
              f"({mem.percent:.1f}% used, {mem.available / 1e9:.1f} GB free)")
    except ImportError:
        print("System RAM: psutil not installed")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                  f"{allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total")


def cleanup() -> None:
    """Collect Python garbage and hand cached blocks back to the CUDA allocator.

    ``empty_cache`` alone is not enough - PyTorch will not release a block that
    a dead-but-uncollected Python object still references, so the ``gc`` pass
    has to come first.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
