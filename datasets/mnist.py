import grain
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from torchvision.datasets import MNIST


def make_mnist_dataset(
    train: bool = True,
    flatten: bool = False,
    onehot: bool = False,
) -> grain.MapDataset:
    """Load the MNIST database of handwritten digits.

    Args:
        train (`bool`, optional): Whether to load the train or test set. Defaults to `True`.
        flatten (`bool`, optional): Whether to flatten images. Defaults to `False`.
        onehot (`bool`, optional): Whether to one-hot encode the labels. Defaults to `False`.

    Returns:
        The MNIST database of handwritten digits.
    """

    def to_numpy(data: tuple) -> tuple:
        return np.array(data[0]), np.array(data[1])

    @jax.jit
    def to_normalized(data: tuple) -> tuple:
        x, y = jnp.array(data[0]), jnp.array(data[1])
        x = x[jnp.newaxis, :] / 255.0
        return x, y

    @jax.jit
    def to_flattened(data: tuple) -> tuple:
        return data[0].ravel(), data[1]

    @jax.jit
    def to_onehot(data: tuple) -> tuple:
        return data[0], jax.nn.one_hot(data[1], num_classes=10)

    dataset = MNIST(root='./data', train=train, download=True)
    return (
        grain.MapDataset.source(dataset)  # type: ignore
        .map(to_numpy)
        .map(to_normalized)
        .map(to_flattened if flatten else lambda x: x)
        .map(to_onehot if flatten else lambda x: x)
    )
