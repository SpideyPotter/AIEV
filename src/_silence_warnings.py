"""Silence noisy deprecation/future warnings from torch / timm / Mask2Former.

Import at the very top of any script:
    import src._silence_warnings  # noqa: F401  (must be first)

These warnings fire on every iter / every import and bury real errors.
What's silenced:
- `Tensor.type() is deprecated` — Mask2Former's CUDA op uses old API
- `Importing from timm.models.layers is deprecated` — timm 1.0 deprecation
- `torch.cuda.amp.{GradScaler,autocast} is deprecated` — torch 2.5+ rename
- `torch.load with weights_only=False` — security hardening notice
- `torch.meshgrid: indexing argument` — torch upgrade migration

Real errors / warnings remain visible (DeprecationWarning is silenced but
non-deprecation UserWarnings still surface).
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
warnings.filterwarnings("ignore", category=UserWarning, module="timm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fvcore.*")

# Belt-and-suspenders: env vars that take effect before warning filters in
# child processes (e.g. dataloader workers).
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::DeprecationWarning,ignore::FutureWarning",
)
os.environ.setdefault("TORCH_WARN_ONCE", "1")
