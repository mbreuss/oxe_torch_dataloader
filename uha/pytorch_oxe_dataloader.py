"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from uha.data.utils.data_utils import hydra_get_object


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
        self._key_remapping = None
        self._combine_goal_obs = False
        self._move_axis = False
        self._add_empty_key = []
        self._adjust_type = None
        self._bytes_to_string = True

        if transform_dict is not None:
            self._key_remapping = transform_dict["key_remapping"]
            self._combine_goal_obs = transform_dict["combine_goal_obs"]
            self._move_axis = transform_dict["move_axis"]
            self._add_empty_key = transform_dict["add_empty_key"]
            self._adjust_type = transform_dict["adjust_type"]
            self._bytes_to_string = transform_dict["bytes_to_string"]

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            if self._combine_goal_obs:
                # print(sample["task"]["image_primary"].shape) => (height, width, channels), (256, 256, 3)
                # print(sample["observation"]["image_primary"].shape) => (window_size, height, width, channels), (2, 256, 256, 3)
                sample["observation"]["image_primary"] = np.concatenate((sample["observation"]["image_primary"], [sample["task"]["image_primary"]]), axis=0)
                sample["observation"]["image_wrist"] = np.concatenate((sample["observation"]["image_wrist"], [sample["task"]["image_wrist"]]), axis=0)
    
            if self._move_axis:
                sample["observation"]["image_primary"] = np.moveaxis(sample["observation"]["image_primary"], 3, 1)
                sample["observation"]["image_wrist"] = np.moveaxis(sample["observation"]["image_wrist"], 3, 1)

            if self._bytes_to_string:
                sample["task"]["language_instruction"] = np.array(sample["task"]["language_instruction"].decode("utf-8"))
            
            if self._adjust_type is not None:
                dtype = hydra_get_object(self._adjust_type)
                sample["observation"]["image_primary"] = torch.from_numpy(sample["observation"]["image_primary"]).to(dtype=dtype)
                sample["observation"]["image_wrist"] = torch.from_numpy(sample["observation"]["image_wrist"]).to(dtype=dtype)
                sample["action"] = torch.from_numpy(sample["action"]).to(dtype=dtype)
            
            # moved _key_remapping into transform_sample
            yield self.transform_sample(sample)

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
        if self._key_remapping is None:
            if len(self._add_empty_key) != 0:
                for key in self._add_empty_key:
                    sample[key] = {}
            return sample
        else:
            transformed_sample = {}
            if len(self._add_empty_key) != 0:
                for key in self._add_empty_key:
                    transformed_sample[key] = {}
            # { observation: { image_primary: ["rgb_obs", "rgb_static"], ... }, ...}
            for old_key, value in self._key_remapping.items():
                if isinstance(value, dict):
                    for second_old_key, new_value in value.items():
                        if isinstance(new_value, list) and len(new_value) == 2:
                            transformed_sample[new_value[0]][new_value[1]] = sample[old_key][second_old_key]
                        elif isinstance(new_value, list) and len(new_value) == 1:
                            transformed_sample[new_value[0]] = sample[old_key][second_old_key]
                        else:
                            transformed_sample[new_value] = sample[old_key][second_old_key]
                else:
                    if isinstance(value, list) and len(value) == 2:
                        transformed_sample[value[0]][value[1]] = sample[old_key]
                    elif isinstance(value, list) and len(value) == 1:
                        transformed_sample[value[0]] = sample[old_key]
                    else:
                        transformed_sample[value] = sample[old_key]

            return transformed_sample
