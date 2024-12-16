import os
# Set environment variables before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_DEBUG"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import time
import hydra
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
import torch
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow
from omegaconf import DictConfig, OmegaConf
import logging
from uha.data.utils.dataset_diagnostics import diagnose_dataset_loading, test_dataset_loading, analyze_transform_compatibility
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='The given NumPy array is not writable')

def setup_gpu():
    """Configure specific GPU and log device information."""
    # TensorFlow GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU 0
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"TensorFlow found {len(gpus)} GPU(s)")
            logger.info(f"Using GPU: {tf.config.list_physical_devices('GPU')[0]}")
            
            # Log GPU memory info
            gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
            logger.info(f"GPU memory info: {gpu_mem}")
        except RuntimeError as e:
            logger.warning(f"Error setting up TensorFlow GPU: {e}")
    else:
        logger.warning("TensorFlow: No GPUs found, using CPU")

    # PyTorch GPU setup
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set PyTorch to use GPU 0
        logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"PyTorch GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        logger.warning("PyTorch: No GPUs found, using CPU")

def optimize_dataset_config(cfg):
    """Optimize dataset configuration for performance."""
    if "interleaved_dataset_cfg" in cfg:
        # Optimize buffer sizes and thread counts
        cfg.interleaved_dataset_cfg.shuffle_buffer_size = 100
        cfg.interleaved_dataset_cfg.traj_transform_threads = 2
        cfg.interleaved_dataset_cfg.traj_read_threads = 2
        
        if "frame_transform_kwargs" in cfg.interleaved_dataset_cfg:
            transform_kwargs = cfg.interleaved_dataset_cfg.frame_transform_kwargs
            transform_kwargs.num_parallel_calls = 2
            
            # Ensure resize configurations exist
            if "resize_size" in transform_kwargs:
                for view in transform_kwargs.resize_size:
                    transform_kwargs.resize_size[view] = [224, 224]
            
            if "resize_size_future_obs" in transform_kwargs:
                for view in transform_kwargs.resize_size_future_obs:
                    transform_kwargs.resize_size_future_obs[view] = [112, 112]
            
            # Minimal image augmentations
            transform_kwargs.image_augment_kwargs = {}
    
    logger.info("Optimized configuration:")
    logger.info(OmegaConf.to_yaml(cfg.interleaved_dataset_cfg))
    
    return cfg

@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    try:
        # Run diagnostics first
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            cfg.DATA_NAME,
            cfg.DATA_PATH,
            load_camera_views=cfg.load_camera_views,
        )
        
        for dataset_kwargs in dataset_kwargs_list:
            # Run diagnostics
            diagnostic_results = diagnose_dataset_loading(dataset_kwargs)
            if diagnostic_results["has_critical_issues"]:
                logger.error("Critical issues found in dataset configuration")
                continue
                
            # Check transform compatibility
            warnings = analyze_transform_compatibility(dataset_kwargs)
            for warning in warnings:
                logger.warning(warning)
            
            # Test dataset loading
            test_dataset_loading(dataset_kwargs)
        logger.info("Starting dataset loading process...")
        
        # Setup specific GPU
        setup_gpu()
        
        # Optimize configuration
        cfg = optimize_dataset_config(cfg)
        
        if "HOME" in cfg:
            os.environ["HOME"] = cfg.HOME
        
        # Load dataset
        logger.info("Loading TensorFlow dataset...")
        dataset = get_octo_dataset_tensorflow(cfg, train=True)
        
        # Add prefetch with controlled buffer size
        dataset = dataset.prefetch(buffer_size=2)
        
        # Test dataset iteration
        logger.info("Testing TensorFlow dataset iteration...")
        try:
            dataset_iter = iter(dataset)
            first_item = next(dataset_iter)
            logger.info("Successfully retrieved first item")
            
            # Log structure
            logger.info("Dataset structure:")
            def log_structure(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, tf.Tensor)):
                            logger.info(f"{prefix}{key}:")
                            log_structure(value, prefix + "  ")
                        else:
                            logger.info(f"{prefix}{key}: {type(value)}")
                elif isinstance(obj, tf.Tensor):
                    logger.info(f"{prefix}Shape: {obj.shape}, dtype: {obj.dtype}")
                    
            log_structure(first_item)
            
        except Exception as e:
            logger.error(f"Error in dataset iteration: {e}")
            raise
            
        # Setup parameters
        is_single_dataset = False
        batch_size = 8  # Modest batch size
        cfg_transforms = OmegaConf.to_object(cfg.transforms)
        
        # Initialize language encoder
        logger.info("Initializing language encoder...")
        language_encoder = hydra.utils.instantiate(cfg.language_encoders)
        
        # Create PyTorch dataloader
        logger.info("Creating PyTorch dataloader...")
        dataloader = make_pytorch_oxe_iterable_dataset(
            dataset, 
            train=True, 
            batch_size=batch_size,
            transform_dict=cfg_transforms,
            num_workers=2,  # Small number of workers
            pin_memory=True,  # Enable pin_memory for better GPU transfer
            language_encoder=language_encoder,
            is_single_dataset=is_single_dataset,
            main_process=True
        )
        
        # Test dataloader
        generator = iter(dataloader)
        first_batch = next(generator)
        logger.info("Successfully got first batch")
        logger.info(f"Batch keys: {first_batch.keys()}")
        logger.info(f"Task keys: {first_batch['task'].keys()}")
        
        # Main iteration loop
        logger.info("Starting main iteration loop...")
        for step in tqdm(range(10)):  # Increased to 10 iterations
            sample = next(generator)
            if step % 2 == 0:  # Log every other step
                logger.info(f"Successfully completed step {step}")
                # Log GPU memory usage during iteration
                if torch.cuda.is_available():
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main()