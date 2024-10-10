"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import torch
import torch.nn as nn
from uha.data.utils.data_utils import hydra_get_object
# from uha.data.utils.lang_buffer import AdvancedLangEmbeddingBuffer
from uha.data.language_encoders.no_encoder import NoEncoder
from dlimp.dataset import DLataset


def print_tensor_shapes(data, prefix=''):
    if isinstance(data, dict):
        for key, value in data.items():
            print_tensor_shapes(value, prefix + key + '.')
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print_tensor_shapes(item, prefix + f'[{i}].')
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        print(f"{prefix[:-1]}: shape {data.shape}")
    else:
        print(f"{prefix[:-1]}: type {type(data)}")


class TorchRLDSIterableDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
            is_single_dataset: bool = False,
            max_cache_size: int = 10000,
    ):
        super(TorchRLDSIterableDataset).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._is_single_dataset = is_single_dataset
        self._current_length = 0
        self._key_remapping = transform_dict["key_remapping"] if transform_dict is not None and "key_remapping" in transform_dict else None
        self._move_axis = transform_dict["move_axis"] if transform_dict is not None and "move_axis" in transform_dict else True
        self._add_empty_key = transform_dict["add_empty_key"] if transform_dict is not None and "add_empty_key" in transform_dict else []
        self._adjust_type = transform_dict["adjust_type"] if transform_dict is not None and "adjust_type" in transform_dict else None
        self._bytes_to_string = transform_dict["bytes_to_string"] if transform_dict is not None and "bytes_to_string" in transform_dict else True
        self._add_robot_information = transform_dict["add_robot_information"] if transform_dict is not None and "add_robot_information" in transform_dict else False

        # self._lang_embedding_buffer = AdvancedLangEmbeddingBuffer(self._language_encoder, goal_instruction_buffer_size=max_cache_size)

    def __iter__(self):
        for sample in self._rlds_dataset.iterator(prefetch=2048): # batchsize
        # for sample in self._rlds_dataset.as_numpy_iterator():
            if self._is_single_dataset: # yield only 1 element of trajectorie
                self._current_length = sample["action"].shape[0]
                for i in range(self._current_length):
                    sub_batch = self.limit_size(sample, dict(), i)
                    yield self.remap_sample(self.transform_sample(sub_batch))

            else:
                sample = self.transform_sample(sample)
                # moved _key_remapping into transform_sample
                yield self.remap_sample(sample)

    def __len__(self):
        if hasattr(self._rlds_dataset, "dataset_len"):
            # print("dataset_len called", self._rlds_dataset.dataset_len)
            return self._rlds_dataset.dataset_len
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths = np.array(self._rlds_dataset.sample_weights) * lengths
        total_len = lengths.sum()
        # print("num_transitions called", total_len)
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

    def limit_size(self, sample, sub_batch, index):
        if isinstance(sample, np.ndarray):
            return sample[index] # if index <= self._current_length-1 else (None, sample[:])
        else:
            for key in sample:
                sub_batch[key] = self.limit_size(sample[key], sub_batch[key] if key in sub_batch else dict(), index)
            return sub_batch

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

        if self._adjust_type is not None:
            dtype = hydra_get_object(self._adjust_type)
            sample["action"] = sample["action"].astype(dtype)

        if self._bytes_to_string:
                if sample["task"]["pad_mask_dict"]["language_instruction"]:
                    sample["task"]["language_instruction"] = sample["task"]["language_instruction"].decode("utf-8")
                    sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
                    # sample["task"]["language_instruction"] = self._lang_embedding_buffer.get_goal_instruction_embeddings(sample["task"]["language_instruction"])
                else:
                    sample["task"]["language_instruction"] = self._lang_embedding_buffer.get_goal_instruction_embeddings("")

        if "robot_information" in sample["task"]:
            if self._add_robot_information:
                # Decode the byte string to a regular string
                robot_info_str = sample["task"]["robot_information"].decode('utf-8')
                sample["task"]["robot_information"] = self._language_encoder(robot_info_str)
                # self._goal_instruction_buffer
                # sample["task"]["robot_information"] = self._lang_embedding_buffer.get_robot_info_embeddings(robot_info_str)
            else:
                del sample["task"]["robot_information"]

        # print_tensor_shapes(sample)
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


class TorchRLDSIterableDatasetTF(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
            is_single_dataset: bool = False,
    ):
        super(TorchRLDSIterableDatasetTF).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._is_single_dataset = is_single_dataset
        self._current_length = 0

    def __iter__(self):
        rlds_iter = map(self.process_batch, self._rlds_dataset.iterator()) # prefetch=1024
        for sample in rlds_iter: # 4 * batchsize
        # for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        if hasattr(self._rlds_dataset, "dataset_len"):
            # print("dataset_len called", self._rlds_dataset.dataset_len)
            return self._rlds_dataset.dataset_len
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths = np.array(self._rlds_dataset.sample_weights) * lengths
        total_len = lengths.sum()
        # print("num_transitions called", total_len)
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

    def process_batch(self, batch):
        if isinstance(self._language_encoder, NoEncoder):
            batch["task"].pop("language_instruction")
        else:
            batch["task"]["language_instruction"] = self._language_encoder(batch["task"]["language_instruction"].decode("utf-8"))
        del batch["dataset_name"]
        return batch