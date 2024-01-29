"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


class TorchRLDSIterableDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset,
            train=True,
            transform_dict=None
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._transform_dict = transform_dict

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield self.transformed_sample(sample)

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

    def transformed_sample(self, sample):
        if self._transform_dict is not None:
            return sample
        else:
            return sample
