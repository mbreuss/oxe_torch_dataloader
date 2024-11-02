from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Union
from pathlib import Path

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
import oxe_torch_dataloader.uha as uha
from torch.utils.data import DataLoader


class DatasetMode(Enum):
    """Enum for dataset modes to make code more type-safe"""
    TRAIN = auto()
    VALIDATION = auto()
    EVALUATION = auto()


@dataclass
class DataModuleConfig:
    """Configuration for UHA Data Module with improved defaults and validation"""
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: int = 8
    transforms: Dict[str, Any] = field(default_factory=dict)
    language_encoders: Optional[Dict[str, Any]] = None
    data_path: Union[str, Path] = "gs://gresearch/robotics"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.num_workers < 0:
            raise ValueError("Number of workers must be non-negative")
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)


class BaseUhaDataModule:
    """Enhanced base class for UHA data modules"""
    
    def __init__(self, datasets: DictConfig, config: DataModuleConfig):
        self.config = config
        self.datasets_cfg = datasets
        self.transforms = OmegaConf.to_object(config.transforms)
        self.language_encoders = (hydra.utils.instantiate(config.language_encoders) 
                                if config.language_encoders else None)
        self._validate_config()
        
        # Disable GPU for TensorFlow operations
        tf.config.set_visible_devices([], "GPU")
        
    def _validate_config(self) -> None:
        """Validate dataset configuration"""
        required_keys = {"DATA_NAME", "DATA_PATH", "load_camera_views"}
        missing_keys = required_keys - set(self.datasets_cfg.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def _create_dataloader(self, 
                          dataset,
                          mode: DatasetMode,
                          batch_size: Optional[int] = None,
                          is_single_dataset: bool = False,
                          main_process: bool = False) -> DataLoader:
        """Enhanced helper method to create dataloaders with better error handling"""
        try:
            return uha.make_pytorch_oxe_iterable_dataset(
                dataset=dataset,
                train=mode == DatasetMode.TRAIN,
                batch_size=batch_size or self.config.batch_size,
                language_encoder=self.language_encoders,
                transform_dict=self.transforms,
                num_workers=self.config.num_workers if main_process else 0,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
                is_single_dataset=is_single_dataset,
                main_process=main_process
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create dataloader: {str(e)}") from e

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics with proper typing"""
        raise NotImplementedError("Subclasses must implement get_dataset_statistics")


class UhaDataModule(BaseUhaDataModule):
    """Standard UHA data module with improved error handling"""
    
    def __init__(self, datasets: DictConfig, **kwargs):
        super().__init__(datasets, DataModuleConfig(**kwargs))
        self.train_datasets = self._load_dataset(DatasetMode.TRAIN)
        self.val_datasets = self._load_dataset(DatasetMode.VALIDATION)

    def _load_dataset(self, mode: DatasetMode):
        """Helper method to load datasets with error handling"""
        try:
            return uha.get_octo_dataset_tensorflow(
                self.datasets_cfg, 
                train=mode == DatasetMode.TRAIN
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {mode.name} dataset: {str(e)}") from e

    def create_train_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.train_datasets, 
            DatasetMode.TRAIN, 
            main_process=main_process
        )

    def create_val_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.val_datasets, 
            DatasetMode.VALIDATION, 
            main_process=main_process
        )

    def get_dataset_statistics(self) -> Dict[str, Dict]:
        return {
            "train_dataset": self.train_datasets.dataset_statistics,
            "val_dataset": self.val_datasets.dataset_statistics
        }


class UhaDataModuleNoValidation(BaseUhaDataModule):
    """UHA data module for training only"""
    
    def __init__(self, datasets: DictConfig, **kwargs):
        super().__init__(datasets, DataModuleConfig(**kwargs))
        self.train_datasets = uha.get_octo_dataset_tensorflow(self.datasets_cfg, train=True)

    def create_train_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.train_datasets, 
            DatasetMode.TRAIN, 
            main_process=main_process
        )

    def create_val_dataloader(self, main_process: bool = False) -> None:
        return None

    def get_dataset_statistics(self) -> Dict[str, Any]:
        return {
            "train_dataset": self.train_datasets.dataset_statistics,
            "val_dataset": None
        }


class UhaDataModuleSequentialValidation(BaseUhaDataModule):
    """UHA data module with sequential validation and improved configuration"""
    
    def __init__(self, datasets: DictConfig, **kwargs):
        super().__init__(datasets, DataModuleConfig(**kwargs))
        self.train_datasets = uha.get_octo_dataset_tensorflow(self.datasets_cfg, train=True)
        self.val_datasets = uha.get_single_dataset_tensorflow(self.datasets_cfg, train=False)

    def create_train_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.train_datasets, 
            DatasetMode.TRAIN, 
            main_process=main_process
        )

    def create_val_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.val_datasets, 
            DatasetMode.VALIDATION,
            batch_size=1
        )

    def get_dataset_statistics(self) -> Dict[str, Any]:
        return {
            "train_dataset": self.train_datasets.dataset_statistics,
            "val_dataset": self.val_datasets.dataset_statistics
        }


class UhaDataModuleEvaluation(BaseUhaDataModule):
    """UHA data module for evaluation with enhanced error handling"""
    
    def __init__(self, datasets: DictConfig, **kwargs):
        super().__init__(datasets, DataModuleConfig(**kwargs))
        self.train_datasets = uha.get_single_dataset_tensorflow(
            self.datasets_cfg, 
            train=False
        )

    def create_train_dataloader(self, main_process: bool = False) -> DataLoader:
        return self._create_dataloader(
            self.train_datasets,
            DatasetMode.EVALUATION,
            is_single_dataset=True
        )

    def create_val_dataloader(self, main_process: bool = False) -> None:
        return None

    def get_dataset_statistics(self) -> Dict[str, Any]:
        return {
            "train_dataset": self.train_datasets.dataset_statistics,
            "val_dataset": None
        }