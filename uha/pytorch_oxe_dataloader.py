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
from dlimp.dataset import DLataset
from torchvision.transforms.functional import convert_image_dtype


class TorchRLDSIterableDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
            is_single_dataset: bool = False,
            batch_size: int = 128
    ):
        super(TorchRLDSIterableDataset).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._is_single_dataset = is_single_dataset
        self._batch_size = batch_size
        self._sub_batch_size = 0
        self._sub_batch = None
        self._key_remapping = transform_dict["key_remapping"] if transform_dict is not None and "key_remapping" in transform_dict else None
        self._combine_goal_obs = transform_dict["combine_goal_obs"] if transform_dict is not None and "combine_goal_obs" in transform_dict else False
        self._move_axis = transform_dict["move_axis"] if transform_dict is not None and "move_axis" in transform_dict else True
        self._add_empty_key = transform_dict["add_empty_key"] if transform_dict is not None and "add_empty_key" in transform_dict else []
        self._adjust_type = transform_dict["adjust_type"] if transform_dict is not None and "adjust_type" in transform_dict else None
        self._bytes_to_string = transform_dict["bytes_to_string"] if transform_dict is not None and "bytes_to_string" in transform_dict else True

    def __iter__(self):
        for sample in self._rlds_dataset.iterator():
        # for sample in self._rlds_dataset.as_numpy_iterator():
            if self._is_single_dataset: # reshape data to batch
                current_length = sample["action"].shape[0]
                if self._sub_batch_size + current_length < self._batch_size:
                    self._sub_batch_size += current_length
                    if self._sub_batch is None:
                        self._sub_batch = sample
                    else:
                        for key in sample:
                            self._sub_batch[key] = self.add_key(sample[key], self._sub_batch[key])
                else:
                    limit = self._batch_size - self._sub_batch_size
                    if self._sub_batch is None:
                        self._sub_batch = self.limit_size(sample, limit)
                    else:
                        for key in sample:
                            self._sub_batch[key] = self.add_key(sample[key], self._sub_batch[key], limit)
                    self._sub_batch_size = 0
                    yield self.remap_sample(self.transform_sample(self._sub_batch))
                    self._sub_batch = None
                            
            else:
                sample = self.transform_sample(sample)
                # moved _key_remapping into transform_sample
                yield self.remap_sample(sample)

    def __len__(self):
        return self._rlds_dataset.dataset_len
    
    def add_key(self, _sample, sub_batch, limit=None):
        if isinstance(_sample, np.ndarray) and limit is None:
            return np.concatenate([sub_batch, _sample], axis=0)
        elif isinstance(_sample, np.ndarray):
            return np.concatenate([sub_batch, _sample[:limit]], axis=0)
        else:
            for key in _sample:
                sub_batch[key] = self.add_key(_sample[key], sub_batch[key], limit)
            return sub_batch

    def limit_size(self, sample, limit):
        if isinstance(sample, np.ndarray):
            return sample[:limit]
        else:
            for key in sample:
                sample[key] = self.limit_size(sample[key], limit)

    def transform_sample(self, sample):
        dicts = ["observation", "task", "future_obs"]
        if self._move_axis:
            for key in dicts:
                if not key in sample:
                    continue
                if "image_primary" in sample[key]:
                    sample[key]["image_primary"] = np.moveaxis(sample[key]["image_primary"], -1, -3)
                if "image_secondary" in sample[key]:
                    sample[key]["image_secondary"] = np.moveaxis(sample[key]["image_secondary"], -1, -3)
                if "image_wrist" in sample[key]:
                    sample[key]["image_wrist"] = np.moveaxis(sample[key]["image_wrist"], -1, -3)
            
        # if self._combine_goal_obs:
        #     # print(sample["task"]["image_primary"].shape) => (height, width, channels), (256, 256, 3)
        #     # print(sample["observation"]["image_primary"].shape) => (window_size, height, width, channels), (2, 256, 256, 3)
        #     sample["observation"]["image_primary"] = np.concatenate((sample["observation"]["image_primary"], [sample["task"]["image_primary"]]), axis=0)
        #     sample["observation"]["image_wrist"] = np.concatenate((sample["observation"]["image_wrist"], [sample["task"]["image_wrist"]]), axis=0)

        if self._adjust_type is not None:
            dtype = hydra_get_object(self._adjust_type)
            # for key in dicts:
            #     if not key in sample:
            #         continue
            #     if "image_primary" in sample[key]:
            #         sample[key]["image_primary"] = sample[key]["image_primary"].astype(dtype)
            #     if "image_secondary" in sample[key]:
            #         sample[key]["image_secondary"] = sample[key]["image_secondary"].astype(dtype)
            #     if "image_wrist" in sample[key]:
            #         sample[key]["image_wrist"] = sample[key]["image_wrist"].astype(dtype)
            sample["action"] = sample["action"].astype(dtype)

        if self._bytes_to_string:
            if self._is_single_dataset:
                # if sample["task"]["pad_mask_dict"]["language_instruction"][0][0]:
                #     sample["task"]["language_instruction"] = sample["task"]["language_instruction"][0][0].decode("utf-8")
                #     sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
                # else:
                #     sample["task"]["language_instruction"] = self._language_encoder("")
                # sample["task"]["language_instruction"] = self._vectorized_lang_encoder(sample["task"]["language_instruction"])
                sample["task"]["language_instruction"] = self._language_encoder(
                    [s.decode("utf-8") for s in sample["task"]["language_instruction"]]
                )
                    
            else:
                # print(sample["task"]["language_instruction"])
                if sample["task"]["pad_mask_dict"]["language_instruction"]:
                    sample["task"]["language_instruction"] = sample["task"]["language_instruction"].decode("utf-8")
                    sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
                else:
                    sample["task"]["language_instruction"] = self._language_encoder("")

        return sample
    
    def remap_sample(self, sample):
        if self._key_remapping is None:
            if len(self._add_empty_key) != 0:
                for key in self._add_empty_key:
                    sample[key] = {}
            if "dataset_name" in sample:
                del sample["dataset_name"]
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
