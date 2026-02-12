import os
import random
from typing import IO, BinaryIO

import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = dataset.shape[0]
    x, y = [], []

    for _ in range(batch_size):
        i = random.randint(0, sequence_length - context_length - 1)
        j = i + context_length

        x.append(torch.from_numpy(dataset[i: j]).to(device))
        y.append(torch.from_numpy(dataset[i + 1: j + 1]).to(device))

    x = torch.stack(x)
    y = torch.stack(y)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    status = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(status, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    status = torch.load(src)

    model.load_state_dict(status['model'])
    optimizer.load_state_dict(status['optimizer'])
    return status['iteration']
