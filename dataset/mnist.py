from typing import Any

import numpy as np
from torchvision.datasets import MNIST

from .dataloader import DataLoader


def make_mnist_dataset(
    train: bool = True,
    flatten: bool = False,
    binarize: bool = False,
    onehot: bool = False,
) -> DataLoader:
    """Load the MNIST database of handwritten digits.

    Args:
        train (`bool`, optional): Whether to load the train or test set. Defaults to `True`.
        binarize (`bool`, optional): Whether to binarize pixels. Defaults to `False`.
        flatten (`bool`, optional): Whether to flatten images. Defaults to `False`.
        onehot (`bool`, optional): Whether to one-hot encode the labels. Defaults to `False`.

    Returns:
        The MNIST database of handwritten digits.
    """

    def transform(x: Any) -> np.ndarray:
        x = np.array(x)[np.newaxis] / 255.0
        x = (x > 0.5).astype(x.dtype) if binarize else x
        return np.ravel(x) if flatten else x

    def target_transform(y: int) -> np.ndarray:
        return np.identity(10)[y] if onehot else np.array(y)

    dataset = MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    return DataLoader(dataset)
