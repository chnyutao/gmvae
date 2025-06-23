import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jax import lax
from jaxtyping import PRNGKeyArray

from .dataloader import DataLoader


def make_random_walks(
    n: int,
    length: int,
    *,
    eps: float = 0.0,
    key: PRNGKeyArray,
) -> DataLoader:
    """Generate a dataset contain random walk trajectories and actions.

    A random walk trajectory is genrated following these rules:
    - An agent lives in a 2D grid world and starts at the origin (0, 0);
    - At any time step t, the agent can move w/a/s/d;
    - The consequence of each action is stochastic.
      Specifically, facing the direction of the given action, the agent may
        - move forward 1 units with probability 0.7;
        - move forward 2 units with probability 0.3.
      Futhermore, the final destination is pertubed by a small Gaussian noise.

    Args:
        n (`int`): The number of trajectories (> 0).
        length (`int`): The length of each trajectory (> 1).
        eps (`float`, optional): Gaussian nosie scale. Default to 0.0.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A dataset containing `n * (length - 1)` transitions, where each transition is a
        3-tuple `(state, action, next_state)`. Each state is a 2d vector `[x y]` and
        each action is a one-hot vector.
    """
    key1, key2, key3 = jr.split(key, 3)
    # generate actions
    action_space = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    indices = jr.randint(key1, (n, length - 1), minval=0, maxval=4)
    actions = action_space[indices]  # (n, length - 1, 2)
    # generate noisy actions
    noise = jr.categorical(key2, jnp.array([0.5, 0.5]), shape=(n, length - 1))
    noisy_actions = jax.vmap(lax.switch, in_axes=(0, None, 0))(
        noise.ravel(),
        [
            lambda action: action,
            lambda action: action * 2,
        ],
        actions.reshape(-1, 2),
    ).reshape(n, length - 1, 2)
    noisy_actions += eps * jr.normal(key3, (n, length - 1, 2))
    # generate states
    states = jnp.cumsum(noisy_actions, axis=1)  # (n, length - 1, 2)
    states = jnp.concat((jnp.zeros((n, 1, 2)), states), axis=1)  # (n, length, 2)
    # generate one-hot actions
    actions = jax.nn.one_hot(indices, 4)  # (n, length - 1, 4)
    actions = jnp.concat((actions, jnp.zeros((n, 1, 4))), axis=1)  # (n, length, 4)
    # return
    next_states = states[:, 1:].reshape(-1, 2)
    states = states[:, :-1].reshape(-1, 2)
    actions = actions[:, :-1].reshape(-1, 4)
    return DataLoader(jdl.ArrayDataset(states, actions, next_states, asnumpy=False))
