import os
import torch

def setup_device_and_parallel(model: torch.nn.Module, prefer_bf16_on_cuda: bool = True):
    """
    Picks the best available device across Windows/Linux/Mac and enables multi-GPU on CUDA.
    Returns (device_str, model_wrapped, autocast_dtype).

    - CUDA: uses all GPUs via DataParallel if >1 GPU.
    - MPS (Apple Silicon): uses "mps" (single device; Apple doesn't expose multi-GPU in PyTorch).
    - CPU: fallback.

    autocast_dtype is the dtype you should pass to torch.autocast if you choose to use AMP.
    """

    # Allow unsupported MPS ops to fall back to CPU automatically (safe no-op elsewhere).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    autocast_dtype = None

    if torch.cuda.is_available():
        device = "cuda"
        n = torch.cuda.device_count()
        if n > 1:
            print(f"[INFO] Using CUDA with {n} GPUs via DataParallel.")
            model = torch.nn.DataParallel(model)
        else:
            print("[INFO] Using CUDA with 1 GPU.")

        # Speed knobs for conv-heavy models
        torch.backends.cudnn.benchmark = True

        # Prefer bf16 if supported, else fp16; leave None to disable AMP
        if prefer_bf16_on_cuda and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16

    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
        print("[INFO] Using Apple Metal (MPS).")
        autocast_dtype = None  # AMP for MPS is still limited/unreliable for detection models

    else:
        device = "cpu"
        print("[INFO] Using CPU.")
        autocast_dtype = None

    return device, model.to(device), autocast_dtype
