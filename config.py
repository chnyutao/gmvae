from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class Config:
    batch_size: int = 64
    """Batch size."""

    beta: tuple[float, float] = (1, 1)
    """Weights for categorical/Gaussian KL divergences."""

    epochs: int = 50
    """Epochs."""

    independent: bool = True
    """Gaussian mixture encoder output specification.
    - If `True`, output k Gaussians independent given x;
    - If `False`, output one Gaussian dependent on x and y.
    """

    k: int = 10
    """Number of Gaussians components."""

    latent_size: int = 32
    """Dimensionality of each Gaussian."""

    likelihood: Literal['gaussian', 'bernoulli'] = 'gaussian'
    """Likelihood p(x|z), Gaussian N(x_hat,I) or Bernoulli Ber(x_hat)."""

    lr: float = 1e-4
    """Learning rate."""

    prior: Literal['standard', 'conditional'] = 'conditional'
    """Gaussian prior p(z|y), standard N(0,I) or conditional N(m,s)=f(y)."""

    sampling: Literal['gumbel', 'st', 'both'] = 'gumbel'
    """Sampling method for discrete latent variables.
    - If `gumbel`, use gumbel-softmax approximation `y = gumbel-softmax(probs)`;
    - If `st`, use straight-through gradient estimation `y = sample + probs - sg(probs)`;
    - If `both`, use straight-through gumbel-softmax trick `y = sample + y_soft - sg(y_soft)`."""

    seed: int = 42
    """Random seed."""

    tau: float = 1.0
    """Gumbel softmax temperature."""

    def asdict(self) -> dict:
        return asdict(self)
