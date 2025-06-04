from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

import equinox as eqx
import jax
import jax.random as jr
from equinox import nn
from jaxtyping import PRNGKeyArray


class MLP(nn.Sequential):
    """Multi-layer perceptron."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        hidden_sizes: Sequence[int] = (),
        activation: Callable = jax.nn.relu,
        key: PRNGKeyArray,
    ):
        """Initializes a multi-layer perceptron (MLP).

        Args:
            in_size (`int`): Input layer ize.
            out_size (`int`): Output layer size.
            hidden_sizes (`Sequence[int]`, optional): Hidden layer sizes.
            activation (`Callable`, optional): Activation function. Defaults to `jax.nn.relu`.
            key (`PRNGKeyArray`): JAX random key.
        """
        layers = []
        layer_sizes = (in_size, *hidden_sizes, out_size)
        keys = iter(jr.split(key, len(layer_sizes) - 1))
        for in_size, out_size in pairwise(layer_sizes):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        super().__init__(layers[:-1])  # remove last activation
