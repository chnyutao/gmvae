from collections.abc import Callable, Iterator
from typing import Self

import jax_dataloader as jdl
import torch.utils.data as torch

Dataset = jdl.Dataset | torch.Dataset


class DataLoader:
    """DataLoader wrapper with function currying."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialize the dataloader.

        Args:
            dataset (`jdl.Dataset | torch.Dataset`): Dataset in jax/torch to be loaded.
        """
        self.collect_fn = None
        self.dataloader = None
        self.dataset = dataset
        self.kwargs = {
            'backend': 'pytorch' if isinstance(dataset, torch.Dataset) else 'jax',
            'batch_size': 1,
            'shuffle': True,
        }

    def batch(self, batch_size: int) -> Self:
        """
        Set the dataloader `batch_size`.

        Args:
            batch_size (`int`): Batch size.

        Returns:
            Dataloader with updated `batch_size`.
        """
        self.kwargs['batch_size'] = batch_size
        return self

    def collect(self, collect_fn: Callable) -> Self:
        """
        Set the collection function, which is used to process a batch of samples
        before they are returned by the data loader

        Args:
            fn (`Callable`): The collection function.

        Returns:
            The dataloader with updated `collect_fn`.
        """
        self.collect_fn = collect_fn
        return self

    def shuffle(self, shuffle: bool = True) -> Self:
        """
        Set the dataloader `shuffle`.

        Args:
            shuffle (`bool`, optional): Whether to shuffle the dataloader each epoch.

        Returns:
            Dataloader with updated `shuffle`.
        """
        self.kwargs['shuffle'] = shuffle
        return self

    def __iter__(self) -> Iterator:
        """
        Return an iterator over elements of the dataloader.

        Returns:
            An iterator for the dataloader.
        """
        if self.dataloader is None:
            self.dataloader = jdl.DataLoader(self.dataset, **self.kwargs)
        if self.collect_fn is None:
            return iter(self.dataloader)
        return (self.collect_fn(batch) for batch in self.dataloader)
