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
            transformed_sample = self.transform_sample(sample)
            transformed_sample["rgb_obs"]["rgb_static"] = torch.from_numpy(np.moveaxis(transformed_sample["rgb_obs"]["rgb_static"], 3, 1)).half()
            transformed_sample["rgb_obs"]["rgb_gripper"] = torch.from_numpy(np.moveaxis(transformed_sample["rgb_obs"]["rgb_gripper"], 3, 1)).half()
            transformed_sample["actions"] = torch.from_numpy(transformed_sample["actions"]).to(transformed_sample["rgb_obs"]["rgb_static"].dtype)
            yield transformed_sample

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

    def transform_sample(self, sample):
        if self._transform_dict is None:
            return sample
        else:
            transformed_sample = {}
            # { observation: { image_primary: ["rgb_obs", "rgb_static"], ... }, ...}
            for old_key, value in self._transform_dict.items():
                if isinstance(value, dict):
                    for second_old_key, new_value in value.items():
                        if isinstance(new_value, list) and len(new_value) == 2:
                            if new_value[0] not in transformed_sample.keys():
                                transformed_sample[new_value[0]] = {}
                            transformed_sample[new_value[0]][new_value[1]] = sample[old_key][second_old_key]
                        elif isinstance(new_value, list) and len(new_value) == 1:
                            transformed_sample[new_value[0]] = sample[old_key][second_old_key]
                        else:
                            transformed_sample[new_value] = sample[old_key][second_old_key]
                else:
                    if isinstance(value, list) and len(value) == 2:
                        if value[0] not in transformed_sample.keys():
                            transformed_sample[value[0]] = {}
                        transformed_sample[value[0]][value[1]] = sample[old_key]
                    elif isinstance(value, list) and len(value) == 1:
                        transformed_sample[value[0]] = sample[old_key]
                    else:
                        transformed_sample[value] = sample[old_key]

            return transformed_sample
