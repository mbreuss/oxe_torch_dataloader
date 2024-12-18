import oxe_torch_dataloader.uha as uha
import hydra

from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
import os 
import copy

class UhaDataModule:
    def __init__(
            self,
            datasets: DictConfig,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            transforms: DictConfig = None,
            language_encoders: DictConfig = None,
            **kwargs,
    ):
        # Store initialization parameters
        self.batch_size = batch_size
        self.train_datasets_cfg = datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Get distributed training information
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"Starting UhaDataModule on rank {self.rank}/{self.world_size}")

        # Initialize transforms and encoders
        print("Creating transforms")
        self.transforms = OmegaConf.to_object(transforms)
        print("Transforms created")
        
        print("Creating language encoders")
        self.language_encoders = hydra.utils.instantiate(language_encoders)
        print("Language encoders created")

        # Configure TensorFlow for distributed training
        tf.config.run_functions_eagerly(True)
        
        # Set different random seeds for different processes
        tf.random.set_seed(42 + self.rank)

        # Initialize datasets with rank-specific configuration
        print(f"Initializing datasets for rank {self.rank}")
        self._initialize_datasets()
        print("UhaDataModule initialization complete")

    def _initialize_datasets(self):
        """Initialize datasets with distributed training considerations."""
        # Modify dataset configuration for distributed training
        distributed_cfg = copy.deepcopy(self.train_datasets_cfg)
        
        # Adjust buffer sizes based on world size to prevent memory issues
        if hasattr(distributed_cfg, 'interleaved_dataset_cfg'):
            distributed_cfg.interleaved_dataset_cfg.shuffle_buffer_size //= self.world_size
            distributed_cfg.interleaved_dataset_cfg.traj_transform_threads //= max(1, self.world_size)
            distributed_cfg.interleaved_dataset_cfg.traj_read_threads //= max(1, self.world_size)

        # Initialize datasets
        print("Creating train dataset")
        self.train_datasets = uha.get_octo_dataset_tensorflow(
            distributed_cfg, train=True)
        print("Train dataset created")
        
        print("Creating validation dataset")
        self.val_datasets = uha.get_octo_dataset_tensorflow(
            distributed_cfg, train=False)
        print("Validation dataset created")

    def create_train_dataloader(self, main_process=False):
        """Create distributed-aware train dataloader."""
        # Calculate workers per GPU
        effective_workers = max(1, self.num_workers // self.world_size) if main_process else 0
        
        return uha.make_pytorch_oxe_iterable_dataset(
            dataset=self.train_datasets,
            train=True,
            batch_size=self.batch_size,
            language_encoder=self.language_encoders,
            transform_dict=self.transforms,
            num_workers=effective_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            main_process=main_process
        )

    def create_val_dataloader(self, main_process=False):
        """Create validation dataloader with distributed considerations."""
        # Validation typically runs on main process only
        effective_workers = max(1, self.num_workers // self.world_size) if main_process else 0
        
        return uha.make_pytorch_oxe_iterable_dataset(
            dataset=self.val_datasets,
            train=False,
            batch_size=self.batch_size,
            language_encoder=self.language_encoders,
            transform_dict=self.transforms,
            num_workers=effective_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            main_process=main_process
        )
    
    def get_dataset_statistics(self):
        """Get dataset statistics with distributed training safety."""
        try:
            return {
                "train_dataset": self.train_datasets.dataset_statistics,
                "val_dataset": self.val_datasets.dataset_statistics
            }
        except AttributeError as e:
            print(f"Warning: Could not access dataset statistics on rank {self.rank}: {str(e)}")
            return {}


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


    def create_train_dataloader(self, main_process=False):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=True, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, main_process=main_process)
    
    def create_val_dataloader(self, main_process=False):
        return None
    
    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": None}


class UhaDataModuleSeqValidationSet:

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
        self.val_datasets = uha.get_single_dataset_tensorflow(self.train_datasets_cfg, train=False)

    def create_train_dataloader(self, main_process=False):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=True, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, main_process=main_process)

    def create_val_dataloader(self, main_process=False):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.val_datasets, train=False, batch_size=1, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)

    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": self.val_datasets.dataset_statistics}


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


    def create_train_dataloader(self, main_process=False):
        return uha.make_pytorch_oxe_iterable_dataset(dataset=self.train_datasets, train=False, batch_size=self.batch_size, language_encoder=self.language_encoders,
                                                     transform_dict=self.transforms, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, is_single_dataset=True)
    
    def create_val_dataloader(self, main_process=False):
        return None
    
    def get_dataset_statistics(self):
        return {"train_dataset": self.train_datasets.dataset_statistics, "val_dataset": None}
