import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tyro
import wandb
from jaxtyping import Array, PRNGKeyArray
from optax import OptState
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
train_set = make_mnist_dataset(train=True, flatten=True)
test_set = make_mnist_dataset(train=False, flatten=True)

# init model
key, cekey, gekey, dkey, pkey = jr.split(key, 5)
model = GMVAE(
    cat_encoder=MLP(784, config.k, hidden_sizes=(200, 200), key=cekey),
    gauss_encoder=MLP(
        784 + (0 if config.encoder == 'wide' else config.k),
        config.latent_size * 2 * (config.k if config.encoder == 'wide' else 1),
        hidden_sizes=(200, 200),
        key=gekey,
    ),
    decoder=MLP(config.latent_size, 784, hidden_sizes=(200, 200), key=dkey),
    prior=(
        MLP(config.k, config.latent_size * 2, key=pkey)
        if config.prior == 'conditional'
        else lambda _: jnp.zeros((config.latent_size * 2,))
    ),
    tau=config.tau,
    wide=config.encoder == 'wide',
)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))


# train/eval functions
@eqx.filter_jit
def train_step(
    model: GMVAE,
    x: Array,
    opt_state: OptState,
    *,
    key: PRNGKeyArray,
) -> tuple[GMVAE, OptState, dict[str, Array]]:
    """
    Perform a single train step for the GMVAE model.

    Args:
        model (`GMVAE`): The current model.
        x (`Array`): Input batched data.
        opt_state (`OptState`): The current optimizer state.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A tuple containing the updated model, the updated optimizer state,
        and a dictionary of training metrics.
    """
    [_, metrics], grads = loss_fn(model, x, config.beta, key=key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, metrics


@eqx.filter_jit
def eval_step(
    model: GMVAE,
    x: Array,
    y: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """
    Compute the clustering purity for the GMVAE model.

    Args:
        model (`GMVAE`): The current model.
        x (`Array`): Input batched data.
        y (`Array`): Input batched labels.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        Purity, a real number between 0.0 and 1.0.
    """
    purity = jnp.array([0])
    _, distributions = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))
    y_hat = distributions['logits'].argmax(axis=-1)
    for k in jnp.arange(config.k):
        _, counts = jnp.unique_counts(jnp.where(y_hat == k, y, jnp.nan), size=10)
        purity += counts.max()
    return purity / x.shape[0]


# main loop
fn = lambda batch: (jax.device_put(batch[0]), jax.device_put(batch[1]))  # noqa: E731
for _ in tqdm(range(config.epochs)):
    # train
    for x, _ in train_set.shuffle().batch(config.batch_size).collect(fn):
        key, subkey = jr.split(key)
        [
            model,
            opt_state,
            metrics,
        ] = train_step(model, x, opt_state, key=subkey)
        wandb.log(metrics)
    # evaluation
    for x, y in test_set.batch(10_000).collect(fn):
        key, subkey = jr.split(key)
        purity = eval_step(model, x, y, key=subkey)
        wandb.log({'purity': purity})
