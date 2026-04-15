"""Train the character RNN on name lists (PyTorch tutorial-style loop)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import build_checkpoint, save_checkpoint
from .data import load_dataset, random_training_pair
from .model import CharRNN
from .paths import DEFAULT_DATA_DIR


def category_from_output(output, all_categories: list[str]):
    top_n, top_i = output.topk(1, dim=1)
    category_i = int(top_i[0, 0].item())
    return all_categories[category_i], category_i


def train_step(
    model: CharRNN,
    category_tensor: torch.Tensor,
    line_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    hidden = model.init_hidden()
    optimizer.zero_grad()
    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, float(loss.item())


def run_training(
    data_dir: Path,
    out_path: Path,
    n_hidden: int,
    n_epochs: int,
    learning_rate: float,
    print_every: int,
    plot_every: int,
    seed: int | None = None,
) -> None:
    if seed is not None:
        torch.manual_seed(seed)
    all_categories, category_lines, all_letters = load_dataset(data_dir)
    n_letters = len(all_letters)
    n_categories = len(all_categories)

    model = CharRNN(
        input_size=n_letters,
        hidden_size=n_hidden,
        output_size=n_categories,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    current_loss = 0.0
    all_losses: list[float] = []
    start = time.time()

    def time_since(since: float) -> str:
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return f"{m:d}m {s:.0f}s"

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = random_training_pair(
            all_categories, category_lines, all_letters
        )
        output, loss = train_step(
            model, category_tensor, line_tensor, optimizer, criterion
        )
        current_loss += loss

        if epoch % print_every == 0:
            guess, _ = category_from_output(output, all_categories)
            correct = "✓" if guess == category else f"✗ ({category})"
            print(
                f"{epoch:d} {epoch / n_epochs * 100:.0f}% "
                f"({time_since(start)}) {loss:.4f} {line} / {guess} {correct}"
            )

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0.0

    payload = build_checkpoint(
        model.state_dict(),
        all_categories,
        all_letters,
        n_hidden,
    )
    save_checkpoint(out_path, payload)
    print(f"Saved checkpoint to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train name-origin CharRNN")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing <Category>.txt name lists",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("weights.pt"),
        help="Checkpoint path (weights + metadata)",
    )
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--print-every", type=int, default=5000)
    p.add_argument("--plot-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    run_training(
        data_dir=args.data_dir,
        out_path=args.output,
        n_hidden=args.n_hidden,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        print_every=args.print_every,
        plot_every=args.plot_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
