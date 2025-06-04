from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import MultivariateNormalDiag, OneHotCategorical
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray, PyTree


class GMEncoder(eqx.Module):
    """Gaussian mixture encoder."""

    cat_encoder: Callable[..., Array]
    gauss_encoder: Callable[..., Array]
    tau: float = eqx.field(static=True)
    wide: bool = eqx.field(static=True)

    def __init__(
        self,
        cat_encoder: Callable[..., Array],
        gauss_encoder: Callable[..., Array],
        *,
        tau: float = 1.0,
        wide: bool = False,
    ):
        """
        Initialize a Gaussian mixture encoder.

        Args:
            cat_encoder (`eqx.Module`): Categorical encoder module.
            gauss_encoder (`eqx.Module`): Gaussian encoder module.
            tau (`float`, optional): Temperature for gumbel-softmax. Defaults to 1.0.
            wide (`bool`, optional): Wide/narrow encoder. Defaults to `True`.
        """
        self.cat_encoder = cat_encoder
        self.gauss_encoder = gauss_encoder
        self.tau = tau
        self.wide = wide

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, Array, PyTree]:
        """
        Forward the input array through the encoder.

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A tuple containing the sampled categorical latent `y`, Gaussian latent `z`, and
            a dictionary of the following format: `{'logits': ..., 'posterior': (mu, std)}`.
        """
        keys = iter(jr.split(key, 3))
        logits = self.cat_encoder(x)
        # sample y ~ q(y|x)
        logits = jax.nn.log_softmax(logits)
        y_soft = jax.nn.softmax((logits + jr.gumbel(next(keys), logits.shape)) / self.tau)
        y = (jnp.arange(*logits.shape) == y_soft.argmax()).astype(int) + y_soft - sg(y_soft)
        # sample z ~ q(z|x,y)
        if self.wide:
            gaussian = (y * self.gauss_encoder(x).reshape(-1, y.shape[0])).sum(axis=-1)
        else:
            gaussian = self.gauss_encoder(jnp.concat((x, y)))
        mu, log_std = jnp.split(gaussian, 2)
        z = mu + jnp.exp(log_std) * jr.normal(next(keys), mu.shape)
        # return
        return y, z, {'logits': logits, 'posterior': (mu, jnp.exp(log_std))}


class GMVAE(eqx.Module):
    """Gaussian mixture variational auto-encoder (GMVAE)."""

    encoder: GMEncoder
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
            **kwargs (`Any`): Extra keyword arguments for `GMEncoder`.
        """
        self.encoder = GMEncoder(
            cat_encoder=cat_encoder,
            gauss_encoder=gauss_encoder,
            **kwargs,
        )
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
        y, z, dists = self.encoder(x, key=key)
        # compute prior p(z|y)
        mu, log_std = jnp.split(self.prior(y), 2)
        # return
        return self.decoder(z), {
            **dists,
            'prior': (mu, jnp.exp(log_std)),
        }


@eqx.filter_jit
@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(
    model: GMVAE, x: Array, betas: tuple[float, float], *, key: PRNGKeyArray
) -> tuple[Array, dict[str, Array]]:
    """negative evidence lower bound (ELBO)."""
    x_hat, dists = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))
    # reconstruction error
    reconst = jnp.sum((x - x_hat) ** 2, axis=-1).mean()
    # categorical entropy
    logits = jax.nn.softmax(dists['logits'])
    nent = jnp.sum(logits * jnp.log(logits), axis=-1).mean()
    # gaussian kld(q(z|x,y) || p(z|y))
    posterior = MultivariateNormalDiag(*dists['posterior'])
    prior = MultivariateNormalDiag(*dists['prior'])
    kld = posterior.kl_divergence(prior).mean()
    # return loss + metrics
    loss = reconst + betas[0] * nent + betas[1] * kld
    return loss, {'loss': loss, 'reconst': reconst, 'entropy': -nent, 'kld': kld}
