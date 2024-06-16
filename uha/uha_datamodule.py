import oxe_torch_dataloader.uha as uha
import hydra

from omegaconf import DictConfig, OmegaConf


class UhaDataModule:

    def __init__(
            self,
            datasets: DictConfig,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            transforms: DictConfig = None,  # Replace with your default transforms
            language_encoders: DictConfig = None,
            **kwargs,
    ):
        self.batch_size = batch_size
        self.train_datasets_cfg = datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.transforms = OmegaConf.to_object(transforms)
        self.language_encoders = hydra.utils.instantiate(language_encoders)
        self.train_datasets = uha.get_octo_dataset_tensorflow(self.train_datasets_cfg, train=True)
        self.val_datasets = uha.get_octo_dataset_tensorflow(self.train_datasets_cfg, train=False)


    def create_train_dataloader(self):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=True, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)

    def create_val_dataloader(self):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.val_datasets, train=False, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)
    
    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": self.val_datasets}


class UhaDataModuleNoValidationSet:

    def __init__(
            self,
            datasets: DictConfig,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            transforms: DictConfig = None,  # Replace with your default transforms
            language_encoders: DictConfig = None,
            **kwargs,
    ):
        self.batch_size = batch_size
        self.train_datasets_cfg = datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.transforms = OmegaConf.to_object(transforms)
        self.language_encoders = hydra.utils.instantiate(language_encoders)
        self.train_datasets = uha.get_octo_dataset_tensorflow(self.train_datasets_cfg, train=True)


    def create_train_dataloader(self):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=True, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)
    
    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": None}


class UhaDataModuleEvaluation:

    def __init__(
            self,
            datasets: DictConfig,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            transforms: DictConfig = None,  # Replace with your default transforms
            language_encoders: DictConfig = None,
            **kwargs,
    ):
        self.batch_size = batch_size
        self.train_datasets_cfg = datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.transforms = OmegaConf.to_object(transforms)
        self.language_encoders = hydra.utils.instantiate(language_encoders)
        self.train_datasets = uha.get_single_dataset_tensorflow(self.train_datasets_cfg, train=False)


    def create_train_dataloader(self):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=False, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)
    
    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": None}
