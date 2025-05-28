from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class Config:
    batch_size: int = 64
    """Batch size."""

    beta: tuple[int, int] = (1, 1)
    """Weights for categorical/Gaussian KL divergences."""

    epochs: int = 20
    """Epochs."""

    k: int = 10
    """Number of Gaussians components."""

    latent_size: int = 32
    """Dimensionality of each Gaussian."""

    lr: float = 1e-4
    """Learning rate."""

    prior: Literal['standard'] | Literal['conditional'] = 'conditional'
    """Gaussian prior p(z|y), standard N(0,I) or conditional N(m,s)=f(y)."""

    sampling: Literal['st'] | Literal['gumbel'] = 'gumbel'
    """Sampling method for discrete latent variables, straight-through or gumbel-softmax."""

    seed: int = 42
    """Random seed."""

    tau: float = 1.0
    """Gumbel softmax temperature."""

    def asdict(self) -> dict:
        return asdict(self)
