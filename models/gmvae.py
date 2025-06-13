from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import Bernoulli, MultivariateNormalDiag, OneHotCategorical
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray, PyTree


class CatEncoder(eqx.Module):
    """Categorical encoder."""

    nn: Callable[..., Array]
    sampling: str = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    def __init__(self, nn: Callable[..., Array], **kwargs: Any):
        """
        Initialize a categorical encoder.

        Args:
            nn (`Callable[..., Array]`): Neural network.
            **kwargs (`Any`, optional): Extra keyword arguments `sampling` and `tau`.
        """
        self.nn = nn
        self.sampling = kwargs.get('sampling', 'gumbel')
        self.tau = kwargs.get('tau', 1.0)

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """
        Forward the input array through the categorical encoder.

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A tuple containing the sampled categorical latent `y`, and
            a dictionary of the following format: `{'logits': ...}`.
        """
        logits = jax.nn.log_softmax(self.nn(x))
        # sample y ~ q(y|x)
        match self.sampling:
            case 'gumbel':
                y = jax.nn.softmax((logits + jr.gumbel(key, logits.shape)) / self.tau)
            case 'st':
                probs = jnp.exp(logits)
                y = OneHotCategorical(logits).sample(seed=key) + probs - sg(probs)
            case 'both':
                y_soft = jax.nn.softmax((logits + jr.gumbel(key, logits.shape)) / self.tau)
                y = jnp.zeros_like(logits).at[logits.argmax()].set(1) + y_soft - sg(y_soft)
        # return
        return y, {'logits': logits}


class GaussEncoder(eqx.Module):
    """Gaussian encoder."""

    nn: Callable[..., Array]
    independent: bool = eqx.field(static=True)

    def __init__(self, nn: Callable[..., Array], **kwargs: Any):
        """
        Initialize a Gaussian encoder.

        Args:
            nn (`Callable[..., Array]`): Neural network.
            **kwargs (`Any`, optional): Extra keyword arguments `independent`.
        """
        self.nn = nn
        self.independent = kwargs.get('independent', True)

    def __call__(self, x: Array, y: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """
        Forward the input array through the Gaussian encoder.

        Args:
            x (`Array`): Input array.
            y (`Array`): One-hot array of shape `(k,)`.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A tuple containing the sampled Gaussian latent `z`, and
            a dictionary of the following format: `{'posterior': (mu, std)}`.
        """
        k = y.shape[0]
        if self.independent:
            gaussian = (y * self.nn(x).reshape(-1, k)).sum(axis=-1)
        else:
            gaussian = self.nn(jnp.concat((x, y)))
        # sample z ~ q(.|x,y)
        mu, log_std = jnp.split(gaussian, 2)
        z = mu + jnp.exp(log_std) * jr.normal(key, mu.shape)
        # return
        return z, {'posterior': (mu, jnp.exp(log_std))}


class GMVAE(eqx.Module):
    """Gaussian mixture variational auto-encoder (GMVAE)."""

    cat_encoder: CatEncoder
    gauss_encoder: GaussEncoder
    decoder: Callable[..., Array]
    prior: Callable[..., Array]

    def __init__(
        self,
        cat_encoder: Callable[..., Array],
        gauss_encoder: Callable[..., Array],
        decoder: Callable[..., Array],
        prior: Callable[..., Array],
        **kwargs: Any,
    ):
        """
        Initialize a Gaussian mixture variational auto-encoder (VAE).

        Args:
            cat_encoder (`eqx.Module`): Categorical encoder module.
            gauss_encoder (`eqx.Module`): Gaussian encoder module.
            decoder (`eqx.Module`): Decoder module.
            prior (`eqx.Module`): Prior module.
            **kwargs (`Any`): Extra keyword arguments for `CatEncoder` and `GaussEncoder`.
        """
        self.cat_encoder = CatEncoder(cat_encoder, **kwargs)
        self.gauss_encoder = GaussEncoder(gauss_encoder, **kwargs)
        self.decoder = decoder
        self.prior = prior

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """
        Forward the input array through the model.

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A tuple containing the decoded output array `x_hat`, along with a dictionary
            of the following format: `{'logits': ..., 'prior': (mu, std), 'posterior': (mu, std)}`.
        """
        ykey, zkey = jr.split(key)
        y, logits = self.cat_encoder(x, key=ykey)
        z, posterior = self.gauss_encoder(x, y, key=zkey)
        # compute prior p(z|y)
        mu, log_std = jnp.split(self.prior(y), 2)
        # return
        return self.decoder(z), {
            **logits,
            **posterior,
            'prior': (mu, jnp.exp(log_std)),
        }


@eqx.filter_jit
@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(
    model: GMVAE,
    x: Array,
    *,
    beta: tuple[float, float],
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
    # categorical entropy
    logits = jax.nn.log_softmax(dists['logits'])
    nent = jnp.sum(logits * jnp.exp(logits), axis=-1).mean()
    # gaussian kld(q(z|x,y) || p(z|y))
    posterior = MultivariateNormalDiag(*dists['posterior'])
    prior = MultivariateNormalDiag(*dists['prior'])
    kld = posterior.kl_divergence(prior).mean()
    # return loss + metrics
    loss = reconst + beta[0] * nent + beta[1] * kld
    return loss, {'loss': loss, 'reconst': reconst, 'entropy': -nent, 'kld': kld}
