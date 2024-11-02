"""
Observation transformation implementations for the data pipeline.
These transforms operate on the observation dictionary at a per-frame level.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Callable
import logging

import dlimp as dl
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ImageAugmentConfig:
    """Configuration for image augmentation."""
    augment_order: list
    random_resized_crop: Optional[Dict[str, Any]] = None
    random_brightness: Optional[list] = None
    random_contrast: Optional[list] = None
    random_saturation: Optional[list] = None
    random_hue: Optional[list] = None


class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def decode_image(image: tf.Tensor) -> tf.Tensor:
        """Decode image from string tensor."""
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                return None
            return tf.io.decode_image(image, expand_animations=False)
        return image

    @staticmethod
    def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
        """Resize image to target size."""
        return dl.transforms.resize_image(image, size=size)

    @staticmethod
    def resize_depth_image(depth: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
        """Resize depth image to target size."""
        return dl.transforms.resize_depth_image(depth, size=size)


class ObservationTransformer:
    """Handles observation transformations."""

    def __init__(self):
        self.image_processor = ImageProcessor()

    def augment(self, 
                obs: Dict[str, Any],
                seed: tf.Tensor,
                augment_kwargs: Union[Dict[str, Any], Mapping[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Augment images in observation dictionary."""
        if not hasattr(augment_kwargs, "items"):
            raise ValueError(
                "augment_kwargs must be a dict with keys corresponding to image names, "
                "or a single dict with an 'augment_order' key."
            )

        # Get image names from observation
        image_names = {key[6:] for key in obs if key.startswith("image_")}

        # Handle single augmentation dict case
        if "augment_order" in augment_kwargs:
            augment_kwargs = {name: augment_kwargs for name in image_names}

        # Apply augmentations
        for i, name in enumerate(image_names):
            if name not in augment_kwargs:
                continue
                
            kwargs = augment_kwargs[name]
            logger.debug(f"Augmenting image_{name} with kwargs {kwargs}")
            
            obs[f"image_{name}"] = tf.cond(
                obs["pad_mask_dict"][f"image_{name}"],
                lambda: self._apply_augmentation(obs[f"image_{name}"], kwargs, seed + i),
                lambda: obs[f"image_{name}"]  # skip padding images
            )

        return obs

    def _apply_augmentation(self, image: tf.Tensor, kwargs: Dict[str, Any], seed: tf.Tensor) -> tf.Tensor:
        """Apply augmentation sequence to image."""
        return dl.transforms.augment_image(image, **kwargs, seed=seed)

    def image_dropout(self,
                     obs: Dict[str, Any],
                     seed: tf.Tensor,
                     dropout_prob: float,
                     always_keep_key: Optional[str] = None) -> Dict[str, Any]:
        """Drop out images with specified probability."""
        image_keys = [key for key in obs if key.startswith("image_")]
        if not image_keys:
            return obs

        pad_mask = tf.stack([obs["pad_mask_dict"][key] for key in image_keys])
        shuffle_seed, seed = tf.unstack(tf.random.split(seed))

        # Handle always_keep_key
        if always_keep_key:
            if always_keep_key not in image_keys:
                raise ValueError(f"Specified always_keep_key {always_keep_key} not in image_keys: {image_keys}")
            always_keep_index = tf.constant(image_keys.index(always_keep_key), dtype=tf.int64)
        else:
            always_keep_index = tf.cond(
                tf.reduce_any(pad_mask),
                lambda: tf.random.experimental.stateless_shuffle(tf.where(pad_mask)[:, 0], seed=shuffle_seed)[0],
                lambda: tf.constant(0, dtype=tf.int64)
            )

        # Apply dropout
        rands = tf.random.stateless_uniform([len(image_keys)], seed=seed)
        pad_mask = tf.logical_and(
            pad_mask,
            tf.logical_or(
                tf.range(len(image_keys), dtype=tf.int64) == always_keep_index,
                rands > dropout_prob
            )
        )

        # Update images and pad masks
        for i, key in enumerate(image_keys):
            obs["pad_mask_dict"][key] = pad_mask[i]
            obs[key] = tf.cond(
                pad_mask[i],
                lambda: obs[key],
                lambda: tf.zeros_like(obs[key])
            )

        return obs

    def decode_and_resize(self,
                         obs: Dict[str, Any],
                         resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
                         depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]]) -> Dict[str, Any]:
        """Decode and resize images and depth images."""
        # Handle image names
        image_names = {key[6:] for key in obs if key.startswith("image_")}
        depth_names = {key[6:] for key in obs if key.startswith("depth_")}

        # Convert single size to mapping
        if isinstance(resize_size, tuple):
            resize_size = {name: resize_size for name in image_names}
        if isinstance(depth_resize_size, tuple):
            depth_resize_size = {name: depth_resize_size for name in depth_names}

        # Process RGB images
        for name in image_names:
            if name not in resize_size:
                logger.warning(
                    f"No resize_size provided for image_{name}. "
                    "This will result in 1x1 padding images."
                )

            image = obs[f"image_{name}"]
            if image.dtype == tf.string:
                if tf.strings.length(image) == 0:
                    # Create padding image
                    image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
                else:
                    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)

            elif image.dtype != tf.uint8:
                raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")

            if name in resize_size:
                image = self.image_processor.resize_image(image, size=resize_size[name])
            obs[f"image_{name}"] = image

        # Process depth images
        for name in depth_names:
            if name not in depth_resize_size:
                logger.warning(
                    f"No depth_resize_size provided for depth_{name}. "
                    "This will result in 1x1 padding images."
                )

            depth = obs[f"depth_{name}"]
            if depth.dtype == tf.string:
                if tf.strings.length(depth) == 0:
                    depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32)
                else:
                    depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.float32)[..., 0]

            elif depth.dtype != tf.float32:
                raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}")

            if name in depth_resize_size:
                depth = self.image_processor.resize_depth_image(depth, size=depth_resize_size[name])
            obs[f"depth_{name}"] = depth

        return obs


# Create global transformer instance
OBSERVATION_TRANSFORMER = ObservationTransformer()

# Export transform functions
augment = OBSERVATION_TRANSFORMER.augment
image_dropout = OBSERVATION_TRANSFORMER.image_dropout
decode_and_resize = OBSERVATION_TRANSFORMER.decode_and_resize