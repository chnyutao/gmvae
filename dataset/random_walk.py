from collections.abc import Sequence

import grain
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import PRNGKeyArray


def categorical(shape: Sequence[int], probs: jax.Array, *, key: PRNGKeyArray) -> jax.Array:
    """Sample from a categorical distribution.

    Args:
        shape (`Sequence[int]`): A sequence of non-negative integers representing the result shape.
        probs (`jax.Array`): A 1D array of non-negative real numbers that sum to one.
        key (`PRNGKeyArray`): JAX random key.

    Return:
        An array of shape `shape` containing integers from 1 to K, where K is the length of `probs`.
    """
    cdf = jnp.expand_dims(probs.cumsum(), axis=range(len(shape)))
    return jnp.argmax(cdf > jr.uniform(key, (*shape, 1)), axis=-1)


def make_random_walks(n: int, length: int, *, key: PRNGKeyArray) -> grain.MapDataset:
    """Generate a dataset contain random walk trajectories and actions.

    A random walk trajectory is genrated following these rules:
    - An agent lives in a 2D grid world and starts at the origin (0, 0);
    - At any time step t, the agent can move w/a/s/d;
    - The consequence of each action is stochastic.
      Specifically, facing the direction of the given action, the agent may
        - move forward 1 step with probability 0.5;
        - move forward 2 steps with probability 0.3;
        - move forward 1 step and left 1 step with probability 0.1;
        - move forward 1 step and right 1 step with probability 0.1.

    Args:
        n (`int`): The number of trajectories (> 0).
        length (`int`): The length of each trajectory (> 1).
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A dataset containing `n` trajectories, where each trajectory contains `length` state-action
        pairs. Each state is a 2d vector `[x y]` and each action is a one-hot vector.
    """
    action_key, noise_key = jr.split(key)
    # generate actions
    action_space = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    actions = jr.randint(action_key, (n, length - 1), minval=0, maxval=4)
    actions = action_space[actions]  # (n, length - 1, 2)
    # generate actions + noise
    noise = categorical((n, length - 1), jnp.array([0.5, 0.3, 0.1, 0.1]), key=noise_key)
    noisy_actions = jax.vmap(lax.switch, in_axes=(0, None, 0))(
        noise.ravel(),
        [
            lambda action: action,  # p=0.5
            lambda action: action * 2,  # p=0.3
            lambda action: action + action[::-1],  # p=0.1
            lambda action: action - action[::-1],  # p=0.1
        ],
        actions.reshape(-1, 2),
    ).reshape(n, length - 1, 2)
    # generate trajectories (sequences of states)
    states = jnp.cumsum(noisy_actions, axis=1)  # (n, length - 1, 2)
    # pad (initial) state and (final) actions
    states = jnp.concat((jnp.zeros((n, 1, 2)), states), axis=1)
    actions = jnp.concat((actions, jnp.zeros((n, 1, 2))), axis=1)
    return grain.MapDataset.source(list(zip(states, actions)))
