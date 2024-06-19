"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import torch
import torch.nn as nn
from uha.data.utils.data_utils import hydra_get_object
from uha.data.language_encoders.no_encoder import NoEncoder


class TorchRLDSIterableDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._key_remapping = transform_dict["key_remapping"] if transform_dict is not None and "key_remapping" in transform_dict else None
        self._combine_goal_obs = transform_dict["combine_goal_obs"] if transform_dict is not None and "combine_goal_obs" in transform_dict else False
        self._move_axis = transform_dict["move_axis"] if transform_dict is not None and "move_axis" in transform_dict else True
        self._add_empty_key = transform_dict["add_empty_key"] if transform_dict is not None and "add_empty_key" in transform_dict else []
        self._adjust_type = transform_dict["adjust_type"] if transform_dict is not None and "adjust_type" in transform_dict else None
        self._bytes_to_string = transform_dict["bytes_to_string"] if transform_dict is not None and "bytes_to_string" in transform_dict else True

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            if self._move_axis:
                sample["observation"]["image_primary"] = np.moveaxis(sample["observation"]["image_primary"], 3, 1)
                # sample["observation"]["image_wrist"] = np.moveaxis(sample["observation"]["image_wrist"], 3, 1)
                sample["task"]["image_primary"] = np.moveaxis(sample["task"]["image_primary"], -1, 0)
                # sample["task"]["image_wrist"] = np.moveaxis(sample["task"]["image_wrist"], -1, 0)
            
            if self._combine_goal_obs:
                # print(sample["task"]["image_primary"].shape) => (height, width, channels), (256, 256, 3)
                # print(sample["observation"]["image_primary"].shape) => (window_size, height, width, channels), (2, 256, 256, 3)
                sample["observation"]["image_primary"] = np.concatenate((sample["observation"]["image_primary"], [sample["task"]["image_primary"]]), axis=0)
                # sample["observation"]["image_wrist"] = np.concatenate((sample["observation"]["image_wrist"], [sample["task"]["image_wrist"]]), axis=0)

            if self._adjust_type is not None:
                dtype = hydra_get_object(self._adjust_type)
                sample["observation"]["image_primary"] = sample["observation"]["image_primary"].astype(dtype)
                # sample["observation"]["image_wrist"] = sample["observation"]["image_wrist"].astype(dtype)
                sample["task"]["image_primary"] = sample["task"]["image_primary"].astype(dtype)
                # sample["task"]["image_wrist"] = sample["task"]["image_wrist"].astype(dtype)
                sample["action"] = sample["action"].astype(dtype)

            if self._bytes_to_string:
                if sample["task"]["pad_mask_dict"]["language_instruction"]:
                    sample["task"]["language_instruction"] = sample["task"]["language_instruction"].decode("utf-8")
                    sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
                else:
                    sample["task"]["language_instruction"] = self._language_encoder("")

            # print(sample["task"]["language_instruction"][0][0])
            
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
            lengths = np.array(self._rlds_dataset.sample_weights) * lengths
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
