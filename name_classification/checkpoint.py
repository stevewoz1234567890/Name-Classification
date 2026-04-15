"""Save and load training checkpoints (weights + vocabulary metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch


def build_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    all_categories: List[str],
    all_letters: str,
    n_hidden: int,
) -> Dict[str, Any]:
    return {
        "state_dict": state_dict,
        "all_categories": list(all_categories),
        "all_letters": all_letters,
        "n_hidden": int(n_hidden),
        "n_categories": len(all_categories),
        "n_letters": len(all_letters),
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
