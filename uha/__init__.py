import torch
from uha.data.dataset import make_interleaved_dataset, make_single_dataset
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset, TorchRLDSIterableDatasetTF
from uha.data.utils.data_utils import NormalizationType
import tensorflow as tf

import torch.nn as nn
import dlimp as dl

tf.config.set_visible_devices([], "GPU")

def make_pytorch_oxe_iterable_dataset(dataset: dl.DLataset, 
                                    train=True, 
                                    batch_size=512,
                                    transform_dict=None,
                                    language_encoder=None,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False,
                                    is_single_dataset=False,
                                    main_process=False):
    if language_encoder is not None:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict, 
                                                 language_encoder=language_encoder,
                                                 is_single_dataset=is_single_dataset)
    else:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict,
                                                 is_single_dataset=is_single_dataset)

    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'shuffle': False if is_single_dataset else None,
    }

    if main_process and num_workers > 0:
        # Only add multiprocessing settings if workers > 0
        dataloader_kwargs.update({
            'num_workers': num_workers,
            'prefetch_factor': 2,
            'persistent_workers': True,
        })
    else:
        dataloader_kwargs['num_workers'] = 0

    return DataLoader(torch_itarable, **dataloader_kwargs)
    
def multi_worker_iterable_dataset(dataset: dl.DLataset, language_encoder: nn.Module = None, train=True, batch_size=512, transform_dict=None, num_workers=0, pin_memory=False, drop_last=False, is_single_dataset=False, main_process=False):
    if language_encoder is not None:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict, language_encoder=language_encoder, is_single_dataset=is_single_dataset)
    else:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict, is_single_dataset=is_single_dataset)

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset = torch_itarable

    if main_process:
        return DataLoader(
            torch_itarable,
            batch_size=batch_size,
            num_workers=2, # 1/2 for prefetching, dont increase beyond 2, else we get ddos timeout from gsresearch (1 for HoreKa)
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=8,
            shuffle=False if is_single_dataset else None,
            worker_init_fn=worker_init_fn
        )
    else:
        return DataLoader(
            torch_itarable,
            batch_size=batch_size,
            num_workers=0, # important to keep this to 0 so PyTorch does not mess with the parallelism
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=False if is_single_dataset else None,
        )


def get_octo_dataset_tensorflow(cfg: DictConfig, train: bool):
    if not "action_proprio_normalization_type" in cfg:
        action_proprio_normalization_type = NormalizationType("normal")
    else:
        assert cfg.action_proprio_normalization_type == "normal" or cfg.action_proprio_normalization_type == "bounds", "Error in Config, action_proprio_normalization_type should be \"normal\" or \"bounds\""
        action_proprio_normalization_type = NormalizationType(cfg.action_proprio_normalization_type)

    print("constructing dataset inside get_octo_dataset_tensorflow")
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        action_proprio_normalization_type=action_proprio_normalization_type,
        load_camera_views=cfg.load_camera_views,
        dataset_size_limit=cfg.dataset_size_limit if "dataset_size_limit" in cfg else None,
    )
    
    print("finished constructing dataset inside get_octo_dataset_tensorflow")
    if not train:
        cfg.interleaved_dataset_cfg.shuffle_buffer_size = int(cfg.interleaved_dataset_cfg.shuffle_buffer_size / 100)
    # create instance of interleaved_dataset_cfg for transforms to work
    interleaved_dataset_cfg = OmegaConf.to_object(cfg.interleaved_dataset_cfg)
    print("starting to make dataset interleaved")
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        **interleaved_dataset_cfg,
    )
    print("finished constructing dataset inside get_octo_dataset_tensorflow")
    return dataset

def get_single_dataset_tensorflow(cfg: DictConfig, train: bool):
    if not "action_proprio_normalization_type" in cfg:
        action_proprio_normalization_type = NormalizationType("normal")
    else:
        assert cfg.action_proprio_normalization_type == "normal" or cfg.action_proprio_normalization_type == "bounds", "Error in Config, action_proprio_normalization_type should be \"normal\" or \"bounds\""
        action_proprio_normalization_type = NormalizationType(cfg.action_proprio_normalization_type)
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        action_proprio_normalization_type=action_proprio_normalization_type,
        load_camera_views=cfg.load_camera_views,
    )

    # create instance of interleaved_dataset_cfg for transforms to work
    interleaved_dataset_cfg = OmegaConf.to_object(cfg.interleaved_dataset_cfg)

    print("########################################")
    print("constructing single val dataset:", dataset_kwargs_list[0]["name"])
    print("########################################")
    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs_list[0],
        train=train,
        traj_transform_kwargs=interleaved_dataset_cfg["traj_transform_kwargs"],
        frame_transform_kwargs=interleaved_dataset_cfg["frame_transform_kwargs"],
        # batch_size=1
    )

    return dataset
