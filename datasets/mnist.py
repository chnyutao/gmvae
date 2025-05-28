import grain
import jax
import jax.numpy as jnp
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

    def process_image(image):
        img = jnp.array(image) / 255.0
        return img.ravel() if flatten else img[jnp.newaxis, :]

    def process_label(label):
        return jax.nn.one_hot(label, num_classes=10) if onehot else jnp.array([label])

    dataset = MNIST(root='./data', train=train, download=True)
    return grain.MapDataset.source(dataset).map(  # type: ignore
        lambda data: (process_image(data[0]), process_label(data[1]))
    )
