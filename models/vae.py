from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import Bernoulli, MultivariateNormalDiag
from jaxtyping import Array, PRNGKeyArray, PyTree


class VAE(eqx.Module):
    """Variational auto-ecnodr."""

    encoder: Callable[..., Array]
    decoder: Callable[..., Array]

    def __init__(self, encoder: Callable[..., Array], decoder: Callable[..., Array]):
        """Initialize a variational auto-encoder.

        Args:
            encoder (`Callable[..., Array]`): Encoder module.
            decoder (`Callable[..., Array]`): Decoder module.
        """
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """Forward the input through the model.

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A tuple containing the decoded output array `x_hat`, along with a pair
            `(mu, std)` of the posterior distribution parameters.
        """
        mu, log_std = jnp.split(self.encoder(x), 2)
        z = mu + jnp.exp(log_std) * jr.normal(key, mu.shape)
        return self.decoder(z), (mu, jnp.exp(log_std))


@eqx.filter_jit
@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(
    model: VAE,
    x: Array,
    *,
    beta: float,
    likelihood: str,
    key: PRNGKeyArray,
) -> tuple[Array, dict[str, Array]]:
    """negative evidence lower bound (ELBO)."""
    batch_size = x.shape[0]
    x_hat, dists = jax.vmap(model)(x, key=jr.split(key, batch_size))
    # reconstruction error
    x, x_hat = x.reshape(batch_size, -1), x_hat.reshape(batch_size, -1)
    match likelihood:
        case 'gaussian':
            reconst = jnp.sum((x - x_hat) ** 2, axis=-1).mean()
        case 'bernoulli':
            reconst = -Bernoulli(x_hat).log_prob(x).sum(axis=-1).mean()
    # kld(q(z|x,y) || p(z|y))
    posterior = MultivariateNormalDiag(*dists)
    prior = MultivariateNormalDiag()  # N(0,I)
    kld = posterior.kl_divergence(prior).mean()
    # return loss + metrics
    loss = reconst + beta * kld
    return loss, {'loss': loss, 'reconst': reconst, 'kld': kld}
