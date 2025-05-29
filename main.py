from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import tyro
import wandb
from tqdm.auto import tqdm

from config import Config
from dataset import make_mnist_dataset
from models import GMVAE, MLP
from models.gmvae import loss_fn

# configuration
config = tyro.cli(Config)
key = jr.key(config.seed)
wandb.init(project='gmvae', config=config.asdict())

# init dataset
train_set = make_mnist_dataset(train=True, flatten=True, batch_size=config.batch_size)
test_set = make_mnist_dataset(train=False, flatten=True, batch_size=int(1e4))

# init model
key, ekey, dkey, pkey = jr.split(key, 4)
model = GMVAE(
    k=config.k,
    encoder=MLP(
        784,
        config.k + config.k * config.latent_size * 2,
        hidden_sizes=(200, 200),
        key=ekey,
    ),
    decoder=MLP(
        config.latent_size,
        784,
        hidden_sizes=(200, 200),
        key=dkey,
    ),
    prior=MLP(
        config.k,
        config.latent_size * 2,
        key=pkey,
    )
    if config.prior == 'conditional'
    else lambda _: jnp.zeros((config.latent_size * 2,)),
    sampling=config.sampling,
    tau=config.tau,
)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))


# main loop
for _ in tqdm(range(config.epochs)):
    # train
    for x, _ in train_set:
        x = jax.device_put(x)
        key, subkey = jr.split(key)
        [_, metrics], grads = loss_fn(model, x, config.beta, key=key)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        wandb.log(metrics)
    # test
    for x, y in test_set:
        x, y = jax.device_put(x), jax.device_put(y)
        key, subkey = jr.split(key)
        _, distributions = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))
        y_hat = distributions['logits'].argmax(axis=-1)
        purity = 0
        for cluster in jnp.unique(y_hat):
            _, counts = jnp.unique(y[y_hat == cluster], return_counts=True)
            purity += counts.max()
        wandb.log({'purity': purity / len(test_set)})

# plot
N_PLOTS = 4
x = next(iter(train_set))[0][:N_PLOTS]
x_hat = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))[0].clip(0, 1)
plt.figure(figsize=(4, 2))
for row, (title, x) in enumerate({'Original': x, 'Reconstruction': x_hat}.items()):
    for col in range(N_PLOTS):
        plt.subplot(2, N_PLOTS, row * N_PLOTS + col + 1)
        plt.imshow(x[col].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(f'{title if col == 0 else ""}', fontsize=8)
# plt.show()
wandb.log({'plot': wandb.Image(plt)})
