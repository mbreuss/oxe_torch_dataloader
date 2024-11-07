# In scripts/test_loader.py

"""
Debug script for testing the refactored OXE dataset loader.
"""

import os
import time
from typing import Any, Dict, Union
from enum import Enum, auto
import hydra
import numpy as np
from tqdm.auto import tqdm 
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import dlimp as dl
import tensorflow_datasets as tfds

from uha import (
    make_pytorch_oxe_iterable_dataset,
    get_octo_dataset_tensorflow,
    get_single_dataset_tensorflow
)

logger = logging.getLogger(__name__)

class DatasetMode(Enum):
    """Enum for dataset modes."""
    TRAIN = auto()
    VALIDATION = auto()
    EVALUATION = auto()

class DatasetDebugger:
    """Helper class for debugging dataset loading and processing."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup_environment()
        self.dataset = None
        self.dataloader = None

    def _setup_environment(self):
        """Setup environment variables."""
        if "HOME" in self.cfg:
            os.environ["HOME"] = self.cfg.HOME
        
    def load_dataset(self, single_dataset: bool = False):
        """Load dataset with error handling."""
        try:
            if single_dataset:
                self.dataset = get_single_dataset_tensorflow(
                    self.cfg, 
                    train=True
                ).repeat().unbatch()
                logger.info("Loaded single dataset")
            else:
                self.dataset = get_octo_dataset_tensorflow(
                    self.cfg, 
                    train=True
                )
                logger.info("Loaded interleaved dataset")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def _load_dataset(self, mode: DatasetMode) -> Union[dl.DLataset, None]:
        """Load dataset with proper split handling."""
        try:
            if mode == DatasetMode.TRAIN:
                split = "train"
            else:
                split = "validation"

            # Get builder to check available splits
            builder = tfds.builder(
                self.cfg.DATA_NAME,
                data_dir=str(self.cfg.DATA_PATH)
            )
            
            # Adjust split if validation not available
            if split == "validation" and "validation" not in builder.info.splits:
                logger.warning("No validation split found, using last 5% of train")
                split = "train[95%:]"
            elif split == "train" and "validation" not in builder.info.splits:
                split = "train[:95%]"

            return dl.DLataset.from_rlds(
                builder,
                split=split,
                shuffle=(mode == DatasetMode.TRAIN)
            )
            
        except Exception as e:
            logger.error(f"Failed to load {mode.name} dataset: {str(e)}")
            raise

    def create_dataloader(self, batch_size: int = 1024, is_single_dataset: bool = False):
        """Create PyTorch dataloader with error handling."""
        try:
            cfg_transforms = OmegaConf.to_object(self.cfg.transforms)
            language_encoder = hydra.utils.instantiate(self.cfg.language_encoders)
            
            self.dataloader = make_pytorch_oxe_iterable_dataset(
                dataset=self.dataset,
                train=True,
                batch_size=batch_size,
                transform_dict=cfg_transforms,
                num_workers=0,
                pin_memory=True,
                language_encoder=language_encoder,
                is_single_dataset=is_single_dataset,
                main_process=True
            )
            logger.info("Created PyTorch dataloader")
        except Exception as e:
            logger.error(f"Failed to create dataloader: {str(e)}")
            raise

    def inspect_sample(self, sample: Dict[str, Any]):
        """Print detailed information about a sample."""
        print("\nSample inspection:")
        print("=================")
        
        print("\nTop-level keys:", sample.keys())
        
        if "observation" in sample:
            print("\nObservation contents:")
            for key, value in sample["observation"].items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"- {key}: type={type(value)}")

        if "task" in sample:
            print("\nTask contents:")
            for key, value in sample["task"].items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"- {key}: type={type(value)}")

        if "action" in sample:
            action = sample["action"]
            print(f"\nAction shape: {action.shape}, dtype: {action.dtype}")
            print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

    def run_debug_loop(self, num_steps: int = 50, sleep_time: int = 5):
        """Run debug loop with error handling."""
        try:
            generator = iter(self.dataloader)
            time.sleep(sleep_time)  # Allow time for initialization
            
            print(f"\nStarting debug loop for {num_steps} steps...")
            for step in tqdm(range(num_steps)):
                sample = next(generator)
                
                # Every 10 steps, do a detailed inspection
                if step % 10 == 0:
                    self.inspect_sample(sample)
                    
                # Basic shape checks
                assert "observation" in sample, "Missing observation key"
                assert "task" in sample, "Missing task key"
                assert "action" in sample, "Missing action key"
                
        except Exception as e:
            logger.error(f"Error in debug loop at step {step}: {str(e)}")
            raise


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    """Main debug function."""
    try:
        # Initialize debugger
        debugger = DatasetDebugger(cfg)
        
        # Test single dataset loading
        print("\nTesting single dataset loading...")
        debugger.load_dataset(single_dataset=True)
        debugger.create_dataloader(batch_size=32, is_single_dataset=True)
        debugger.run_debug_loop(num_steps=5)
        
        # Test interleaved dataset loading
        print("\nTesting interleaved dataset loading...")
        debugger.load_dataset(single_dataset=False)
        debugger.create_dataloader(batch_size=1024, is_single_dataset=False)
        debugger.run_debug_loop(num_steps=50)
        
    except Exception as e:
        logger.error(f"Debug session failed: {str(e)}")
        raise
    else:
        print("\nDebug session completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()