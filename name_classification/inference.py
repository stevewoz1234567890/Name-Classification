"""Load a checkpoint and score names (top-k log-probabilities per category)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .checkpoint import load_checkpoint
from .data import line_to_tensor, unicode_to_ascii
from .model import CharRNN


@dataclass
class Prediction:
    label: str
    log_probability: float


class NameClassifier:
    """Thread-safe for inference-only use (eval mode, no grads)."""

    def __init__(self, weights_path: Path, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        ckpt = load_checkpoint(weights_path, map_location=self.device)
        self.all_categories: List[str] = list(ckpt["all_categories"])
        self.all_letters: str = str(ckpt["all_letters"])
        self.n_hidden: int = int(ckpt["n_hidden"])
        self.n_categories: int = int(ckpt["n_categories"])
        self.n_letters: int = int(ckpt["n_letters"])

        self.model = CharRNN(
            input_size=self.n_letters,
            hidden_size=self.n_hidden,
            output_size=self.n_categories,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _tensor_for_name(self, name: str) -> torch.Tensor:
        ascii_name = unicode_to_ascii(name.strip(), allowed=self.all_letters)
        if not ascii_name:
            raise ValueError(
                "Name is empty after ASCII normalization or uses unsupported characters"
            )
        t = line_to_tensor(ascii_name, self.all_letters)
        return t.to(self.device)

    @torch.inference_mode()
    def predict_topk(self, name: str, top_k: int) -> Tuple[str, List[Prediction]]:
        """Return normalized name used for inference and ranked predictions."""
        line_tensor = self._tensor_for_name(name)
        hidden = self.model.init_hidden().to(self.device)
        output = None
        for i in range(line_tensor.size(0)):
            output, hidden = self.model(line_tensor[i], hidden)
        assert output is not None
        k = min(top_k, self.n_categories)
        topv, topi = output.topk(k, dim=1)
        preds: List[Prediction] = []
        for i in range(k):
            logp = float(topv[0, i].item())
            idx = int(topi[0, i].item())
            preds.append(Prediction(label=self.all_categories[idx], log_probability=logp))
        ascii_name = unicode_to_ascii(name.strip(), allowed=self.all_letters)
        return ascii_name, preds


def predict_cli(
    name: str,
    top_n: int,
    weights_file: Path,
) -> Sequence[Prediction]:
    clf = NameClassifier(weights_file)
    _, preds = clf.predict_topk(name, top_n)
    return preds
