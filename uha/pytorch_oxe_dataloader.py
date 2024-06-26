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
    ):
        super(TorchRLDSIterableDataset).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._is_single_dataset = is_single_dataset
        self._key_remapping = transform_dict["key_remapping"] if transform_dict is not None and "key_remapping" in transform_dict else None
        self._combine_goal_obs = transform_dict["combine_goal_obs"] if transform_dict is not None and "combine_goal_obs" in transform_dict else False
        self._move_axis = transform_dict["move_axis"] if transform_dict is not None and "move_axis" in transform_dict else True
        self._add_empty_key = transform_dict["add_empty_key"] if transform_dict is not None and "add_empty_key" in transform_dict else []
        self._adjust_type = transform_dict["adjust_type"] if transform_dict is not None and "adjust_type" in transform_dict else None
        self._bytes_to_string = transform_dict["bytes_to_string"] if transform_dict is not None and "bytes_to_string" in transform_dict else True

    def __iter__(self):
        for sample in self._rlds_dataset.iterator():
        # for sample in self._rlds_dataset.as_numpy_iterator():
            sample = self.transform_sample(sample)
            # moved _key_remapping into transform_sample
            yield self.remap_sample(sample)

    def __len__(self):
        return self._rlds_dataset.dataset_len

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
                if sample["task"]["pad_mask_dict"]["language_instruction"][0][0]:
                    sample["task"]["language_instruction"] = sample["task"]["language_instruction"][0][0].decode("utf-8")
                    sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
                else:
                    sample["task"]["language_instruction"] = self._language_encoder("")
                # print(sample["task"]["language_instruction"][0][0])
                # language_instruction = np.empty_like(sample["task"]["language_instruction"], dtype=sample["action"].dtype)
                # for i in range(len(sample["task"]["pad_mask_dict"]["language_instruction"])):
                #     for j in range(len(sample["task"]["pad_mask_dict"]["language_instruction"][i])):
                #         if sample["task"]["pad_mask_dict"]["language_instruction"][i][j]:
                #             # sample["task"]["language_instruction"][i][j] = sample["task"]["language_instruction"][i][j].decode("utf-8")
                #             language_instruction[i][j] = self._language_encoder(sample["task"]["language_instruction"][i][j].decode("utf-8"))
                #         else:
                #             language_instruction[i][j] = self._language_encoder("")
                # sample["task"]["language_instruction"] = language_instruction
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
