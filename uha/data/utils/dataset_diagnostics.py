import os
import logging
from enum import IntEnum
from typing import Dict, Any, List, Optional

import tensorflow as tf

from uha.data.oxe.oxe_dataset_configs import ActionEncoding, ProprioEncoding

logger = logging.getLogger(__name__)

def diagnose_dataset_loading(dataset_kwargs: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Diagnoses common issues with dataset loading.
    
    Args:
        dataset_kwargs: The dataset configuration dictionary
        verbose: Whether to print diagnostic information
        
    Returns:
        Dict containing:
            - issues: List of potential problems found
            - info: Dict of diagnostic information
            - has_critical_issues: Boolean indicating if any critical issues were found
    """
    issues: List[str] = []
    info: Dict[str, Any] = {}
    
    # Check data directory existence and permissions
    data_dir = dataset_kwargs.get("data_dir", "")
    info["data_dir"] = data_dir
    if not tf.io.gfile.exists(data_dir):
        issues.append(f"Data directory does not exist: {data_dir}")
    elif "gs://" not in data_dir:  # Skip permission check for GCS
        try:
            test_file = os.path.join(data_dir, "test_write")
            with tf.io.gfile.GFile(test_file, "w") as f:
                f.write("test")
            tf.io.gfile.remove(test_file)
        except Exception as e:
            issues.append(f"Permission issue with data directory: {str(e)}")
    
    # Check required keys
    required_keys = ["image_obs_keys", "depth_obs_keys", "proprio_encoding", "action_encoding"]
    for key in required_keys:
        if key not in dataset_kwargs:
            issues.append(f"Missing required key: {key}")
    
    # Validate image keys
    if "image_obs_keys" in dataset_kwargs:
        for view, key in dataset_kwargs["image_obs_keys"].items():
            if key is not None and not isinstance(key, str):
                issues.append(f"Invalid image key for {view}: {key}")
            if key is not None:
                info[f"image_key_{view}"] = key
    
    # Check action encoding compatibility
    if "action_encoding" in dataset_kwargs:
        action_enc = dataset_kwargs["action_encoding"]
        proprio_enc = dataset_kwargs.get("proprio_encoding")
        info["action_encoding"] = action_enc
        info["proprio_encoding"] = proprio_enc
        
        if proprio_enc:
            # Verify encoding compatibility
            if action_enc == ActionEncoding.JOINT_POS and proprio_enc != ProprioEncoding.JOINT:
                issues.append("Mismatched action/proprio encoding: JOINT_POS requires JOINT proprio")
            elif action_enc == ActionEncoding.JOINT_POS_BIMANUAL and proprio_enc != ProprioEncoding.JOINT_BIMANUAL:
                issues.append("Mismatched action/proprio encoding: JOINT_POS_BIMANUAL requires JOINT_BIMANUAL proprio")
    
    # Check standardization function
    if "standardize_fn" in dataset_kwargs:
        try:
            info["standardize_fn"] = str(dataset_kwargs["standardize_fn"])
        except Exception as e:
            issues.append(f"Error inspecting standardize_fn: {str(e)}")
    
    # Check language key patterns
    if "language_key" in dataset_kwargs:
        lang_key = dataset_kwargs["language_key"]
        info["language_key"] = lang_key
        if lang_key and not isinstance(lang_key, str):
            issues.append(f"Invalid language key type: {type(lang_key)}")
        if lang_key and "*" not in lang_key and "groundtruth" not in lang_key:
            issues.append("Language key might be missing wildcard (*) pattern")

    # Check filter functions
    if "filter_functions" in dataset_kwargs:
        filter_fns = dataset_kwargs["filter_functions"]
        info["num_filters"] = len(filter_fns) if isinstance(filter_fns, (list, tuple)) else 0
        if not isinstance(filter_fns, (list, tuple)):
            issues.append("filter_functions should be a sequence")

    # Check dataset size limit
    if "dataset_size_limit" in dataset_kwargs:
        size_limit = dataset_kwargs["dataset_size_limit"]
        info["dataset_size_limit"] = size_limit
        if not isinstance(size_limit, (int, type(None))):
            issues.append("dataset_size_limit should be an integer or None")

    if verbose:
        logger.info("\nDataset Loading Diagnostic Report:")
        logger.info(f"Dataset Name: {dataset_kwargs.get('name', 'Unknown')}")
        logger.info("\nConfiguration:")
        for k, v in info.items():
            logger.info(f"  {k}: {v}")
        logger.info("\nPotential Issues:")
        for issue in issues:
            logger.info(f"  - {issue}")
    
    return {
        "issues": issues,
        "info": info,
        "has_critical_issues": len(issues) > 0
    }

def test_dataset_loading(dataset_kwargs: Dict[str, Any], num_samples: int = 1) -> None:
    """
    Attempts to load a few samples from the dataset to verify basic functionality.
    
    Args:
        dataset_kwargs: The dataset configuration dictionary
        num_samples: Number of samples to try loading
    """
    from uha.data.dataset import make_dataset_from_rlds
    
    logger.info(f"Testing dataset loading for {dataset_kwargs.get('name', 'Unknown')}")
    
    try:
        dataset, stats = make_dataset_from_rlds(**dataset_kwargs, train=True)
        logger.info("Successfully created dataset")
        
        # Try to load some samples
        sample_count = 0
        for sample in dataset.take(num_samples).iterator():
            logger.info(f"\nSample {sample_count + 1} structure:")
            tf.nest.map_structure(
                lambda x: logger.info(f"  {x.shape if hasattr(x, 'shape') else type(x)}"), 
                sample
            )
            sample_count += 1
            
        logger.info(f"\nSuccessfully loaded {sample_count} samples")
        logger.info(f"Dataset statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error testing dataset: {str(e)}", exc_info=True)
        raise

def analyze_transform_compatibility(dataset_kwargs: Dict[str, Any]) -> List[str]:
    """
    Analyzes whether the dataset configuration is compatible with available transforms.
    
    Args:
        dataset_kwargs: The dataset configuration dictionary
        
    Returns:
        List of warnings about potential transform compatibility issues
    """
    from uha.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
    
    warnings = []
    dataset_name = dataset_kwargs.get("name", "")
    
    # Check if dataset has a standardization transform
    if dataset_name not in OXE_STANDARDIZATION_TRANSFORMS:
        warnings.append(f"No standardization transform found for dataset {dataset_name}")
        return warnings
    
    # Get the transform function
    transform_fn = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
    
    # Check image key compatibility
    if "image_obs_keys" in dataset_kwargs:
        image_keys = dataset_kwargs["image_obs_keys"]
        for view, key in image_keys.items():
            if key is not None:
                # Check if transform handles this image key
                if f"image_{view}" not in str(transform_fn.__code__.co_code):
                    warnings.append(f"Transform may not handle image key 'image_{view}'")
    
    # Check proprio compatibility
    if "proprio_obs_key" in dataset_kwargs and dataset_kwargs["proprio_obs_key"]:
        if "proprio" not in str(transform_fn.__code__.co_code):
            warnings.append("Transform may not handle proprio observations")
    
    # Check language key compatibility
    if "language_key" in dataset_kwargs and dataset_kwargs["language_key"]:
        if "language_instruction" not in str(transform_fn.__code__.co_code):
            warnings.append("Transform may not handle language instructions")
    
    return warnings