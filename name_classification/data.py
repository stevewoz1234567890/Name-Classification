"""Load name corpora and character-level preprocessing."""

from __future__ import annotations

import random
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_LETTERS = string.ascii_letters + " .,;'-"


def unicode_to_ascii(s: str, allowed: str = DEFAULT_LETTERS) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in allowed
    )


def read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    out: List[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        asc = unicode_to_ascii(raw)
        if asc:
            out.append(asc)
    return out


def load_dataset(data_dir: Path) -> Tuple[List[str], Dict[str, List[str]], str]:
    """Return sorted category names, lines per category, and shared alphabet string."""
    all_categories: List[str] = []
    category_lines: Dict[str, List[str]] = {}
    for path in sorted(data_dir.glob("*.txt")):
        category = path.stem
        lines = read_lines(path)
        if not lines:
            continue
        all_categories.append(category)
        category_lines[category] = lines
    if not all_categories:
        raise FileNotFoundError(f"No name lists found under {data_dir}")
    letters = DEFAULT_LETTERS
    return all_categories, category_lines, letters


def letter_to_index(letter: str, all_letters: str) -> int:
    idx = all_letters.find(letter)
    if idx < 0:
        raise ValueError(f"Character {letter!r} not in alphabet")
    return idx


def line_to_tensor(line: str, all_letters: str):
    import torch

    n_letters = len(all_letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter, all_letters)] = 1
    return tensor


def random_training_pair(
    all_categories: List[str],
    category_lines: Dict[str, List[str]],
    all_letters: str,
):
    import torch

    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line, all_letters)
    return category, line, category_tensor, line_tensor
