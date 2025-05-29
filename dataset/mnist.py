from typing import Any

import jax_dataloader as jdl
import numpy as np
from torchvision.datasets import MNIST


def make_mnist_dataset(
    train: bool = True,
    flatten: bool = False,
    onehot: bool = False,
    **kwargs: Any,
) -> jdl.DataLoader:
    """Load the MNIST database of handwritten digits.

    Args:
        train (`bool`, optional): Whether to load the train or test set. Defaults to `True`.
        flatten (`bool`, optional): Whether to flatten images. Defaults to `False`.
        onehot (`bool`, optional): Whether to one-hot encode the labels. Defaults to `False`.

    Returns:
        The MNIST database of handwritten digits.
    """

    def transform(x: Any) -> np.ndarray:
        x = np.array(x)[np.newaxis] / 255.0
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
    return jdl.DataLoader(dataset, 'pytorch', shuffle=True, **kwargs)
