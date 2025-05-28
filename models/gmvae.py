from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import MultivariateNormalDiag, OneHotCategorical
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray, PyTree


class GMVAE(eqx.Module):
    """Gaussian mixture variational auto-encoder (GMVAE)."""

    encoder: Callable[..., Array]
    decoder: Callable[..., Array]
    prior: Callable[..., Array]
    k: int = eqx.field(static=True)
    sampling: str = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    def __init__(
        self,
        k: int,
        encoder: Callable[..., Array],
        decoder: Callable[..., Array],
        prior: Callable[..., Array],
        *,
        sampling: str = 'gumbel',
        tau: float = 1,
    ):
        """
        Initialize a Gaussian mixture variational auto-encoder (VAE).

        Args:
            k (`int`): Number of Gaussian mixture components.
            encoder (`eqx.Module`): Encoder module.
            decoder (`eqx.Module`): Decoder module.
            sampling (`Literal['st'] | Literal['gumbel']`, optional):
                use straight-through (`"st"`) or gumbel-softmax (`"gumbel"`) gradient estimator
                for discrete sampling. Defaults to `"gumbel"`.
            tau (`float`, optional): Temperature for gumbel-softmax. Defaults to 1.
        """
        self.k = k
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.sampling = sampling
        self.tau = tau

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        key1, key2 = jr.split(key)
        logits, gaussians = jnp.split(self.encoder(x), [self.k])
        # sample y ~ q(y|x)
        match self.sampling:
            case 'gumbel':
                logits = jax.nn.log_softmax(logits)
                y = jax.nn.softmax((logits + jr.gumbel(key1, logits.shape)) / self.tau)
            case 'st':
                probs = jax.nn.softmax(logits)
                y = OneHotCategorical(logits).sample(seed=key1) + probs - sg(probs)
        # sample z ~ q(z|x,y)
        mu, log_std = jnp.split((gaussians.reshape(-1, self.k) * y).sum(axis=1), 2)
        z = mu + jnp.exp(log_std) * jr.normal(key2, mu.shape)
        # compute prior p(z|y)
        mu0, log_std0 = jnp.split(self.prior(y), 2)
        # return
        return self.decoder(z), {
            'logits': logits,
            'posterior': (mu, jnp.exp(log_std)),
            'prior': (mu0, jnp.exp(log_std0)),
        }


@eqx.filter_jit
@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(
    model: GMVAE, x: Array, betas: tuple[int, int], *, key: PRNGKeyArray
) -> tuple[Array, dict]:
    """negative evidence lower bound (ELBO)."""
    x_hat, distributions = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))
    # reconstruction error
    reconst = jnp.sum((x - x_hat) ** 2, axis=-1).mean()
    # categorical entropy
    logits = jax.nn.softmax(distributions['logits'])
    nent = jnp.sum(logits * jnp.log(logits), axis=-1).mean()
    # gaussian kld(q(z|x,y) || p(z|y))
    posterior = MultivariateNormalDiag(*distributions['posterior'])
    prior = MultivariateNormalDiag(*distributions['prior'])
    kld = posterior.kl_divergence(prior).mean()
    # return loss + metrics
    loss = reconst + betas[0] * nent + betas[1] * kld
    return loss, {'loss': loss, 'entropy': -nent, 'kld': kld}
