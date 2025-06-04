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

    encoder: Literal['wide'] | Literal['narrow'] = 'wide'
    """Encoder outupt width.
    - If `wide`, output k Gaussians simultaenously (`k * latent_size * 2`);
    - If `narrow`, output only one Gaussian (`latent_size * 2`).
    """

    k: int = 10
    """Number of Gaussians components."""

    latent_size: int = 32
    """Dimensionality of each Gaussian."""

    lr: float = 1e-4
    """Learning rate."""

    prior: Literal['standard'] | Literal['conditional'] = 'conditional'
    """Gaussian prior p(z|y), standard N(0,I) or conditional N(m,s)=f(y)."""

    seed: int = 42
    """Random seed."""

    tau: float = 1.0
    """Gumbel softmax temperature."""

    def asdict(self) -> dict:
        return asdict(self)
