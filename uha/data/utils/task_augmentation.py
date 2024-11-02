"""
Task augmentation and instruction processing utilities.
Handles random zero-ing of task specifications and instruction rephrasing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import logging
import pickle
from pathlib import Path

import tensorflow as tf
from huggingface_hub import hf_hub_download

from uha.data.utils.data_utils import to_padding

logger = logging.getLogger(__name__)


@dataclass
class AugmentConfig:
    """Configuration for task augmentation."""
    rephrase_prob: float = 0.3
    keep_image_prob: float = 0.5
    paraphrases_repo: str = ""
    paraphrases_filename: str = ""


class ParaphraseManager:
    """Manages paraphrase operations with caching."""

    def __init__(self, repo: str, filename: str):
        self.repo = repo
        self.filename = filename
        self._rephrase_lookup = None
        self._initialize_paraphrases()

    def _initialize_paraphrases(self):
        """Initialize paraphrase lookup table."""
        try:
            if not self.repo or not self.filename:
                logger.warning("No paraphrase repository specified")
                return

            paraphrase_path = hf_hub_download(
                repo_id=self.repo,
                filename=self.filename,
                repo_type="dataset"
            )
            
            with open(paraphrase_path, "rb") as file:
                paraphrases = pickle.load(file)
                self._rephrase_lookup = self._create_lookup_table(paraphrases)
                
        except Exception as e:
            logger.error(f"Failed to initialize paraphrases: {str(e)}")
            self._rephrase_lookup = None

    def _create_lookup_table(self, dictionary: Dict[str, str]) -> tf.lookup.StaticHashTable:
        """Create TensorFlow lookup table from dictionary."""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys, values,
            key_dtype=tf.string,
            value_dtype=tf.string
        )
        
        return tf.lookup.StaticHashTable(
            initializer,
            default_value=""
        )

    def get_paraphrase(self, text: tf.Tensor) -> tf.Tensor:
        """Get paraphrase for input text."""
        if self._rephrase_lookup is None:
            return text
            
        paraphrase = self._rephrase_lookup.lookup(text)
        return tf.where(
            tf.strings.length(paraphrase) > 0,
            tf.strings.join([text, paraphrase], separator=". "),
            text
        )

    def sample_paraphrase(self, 
                         text: tf.Tensor,
                         rephrase_prob: float) -> tf.Tensor:
        """Sample between original and paraphrased text."""
        if self._rephrase_lookup is None:
            return text

        paraphrased = self.get_paraphrase(text[0])
        split_phrases = tf.strings.split(paraphrased, sep=".")
        num_phrases = tf.shape(split_phrases)[0]
        
        random_index = tf.random.uniform(
            tf.shape(text),
            minval=0,
            maxval=num_phrases,
            dtype=tf.int32
        )
        
        sampled = tf.gather(split_phrases, random_index)
        return tf.where(
            tf.random.uniform(shape=()) < rephrase_prob,
            sampled,
            text
        )


class TaskAugmenter:
    """Handles task augmentation operations."""
    
    def __init__(self, config: AugmentConfig):
        self.config = config
        self.paraphrase_manager = ParaphraseManager(
            config.paraphrases_repo,
            config.paraphrases_filename
        )

    def augment_task(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Apply task augmentation to trajectory."""
        try:
            # Rephrase instructions if available
            if "language_instruction" in trajectory["task"]:
                trajectory = self._rephrase_instruction(trajectory)

            # Apply task conditioning dropout
            trajectory = self._delete_task_conditioning(trajectory)
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error in task augmentation: {str(e)}")
            raise

    def _rephrase_instruction(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Rephrase language instruction."""
        original_language = trajectory["task"]["language_instruction"]
        
        # Check if language key is not empty
        if tf.reduce_all(tf.strings.length(original_language) > 0):
            trajectory["task"]["language_instruction"] = \
                self.paraphrase_manager.sample_paraphrase(
                    original_language,
                    self.config.rephrase_prob
                )
            
        return trajectory

    def _delete_task_conditioning(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop out either goal images or language instruction."""
        if "language_instruction" not in trajectory["task"]:
            return trajectory

        # Find image keys
        image_keys = {
            key for key in trajectory["task"].keys()
            if key.startswith("image_") or key.startswith("depth_")
        }
        
        if not image_keys:
            return trajectory

        # Determine what to keep
        traj_len = tf.shape(trajectory["action"])[0]
        should_keep_images = tf.random.uniform([traj_len]) < self.config.keep_image_prob
        should_keep_images |= ~trajectory["task"]["pad_mask_dict"]["language_instruction"]

        # Process each key
        for key in image_keys | {"language_instruction"}:
            should_keep = should_keep_images if key in image_keys else ~should_keep_images
            
            # Pad out the key
            trajectory["task"][key] = tf.where(
                should_keep,
                trajectory["task"][key],
                to_padding(trajectory["task"][key])
            )
            
            # Update pad mask
            trajectory["task"]["pad_mask_dict"][key] = tf.where(
                should_keep,
                trajectory["task"]["pad_mask_dict"][key],
                tf.zeros_like(trajectory["task"]["pad_mask_dict"][key])
            )

        # Update goal timestep
        trajectory["task"]["timestep"] = tf.where(
            should_keep_images,
            trajectory["task"]["timestep"],
            traj_len - 1
        )

        return trajectory


def create_augmenter(config: Optional[Dict[str, Any]] = None) -> TaskAugmenter:
    """Create task augmenter instance."""
    config = AugmentConfig(**(config or {}))
    return TaskAugmenter(config)


# Convenience functions
def delete_and_rephrase(
        trajectory: Dict[str, Any],
        paraphrases_repo: str,
        paraphrases_filename: str,
        rephrase_prob: float,
        keep_image_prob: float
) -> Dict[str, Any]:
    """Apply both deletion and rephrasing augmentations."""
    config = AugmentConfig(
        rephrase_prob=rephrase_prob,
        keep_image_prob=keep_image_prob,
        paraphrases_repo=paraphrases_repo,
        paraphrases_filename=paraphrases_filename
    )
    
    augmenter = TaskAugmenter(config)
    return augmenter.augment_task(trajectory)


def delete_task_conditioning(
        trajectory: Dict[str, Any],
        keep_image_prob: float
) -> Dict[str, Any]:
    """Apply only deletion augmentation."""
    config = AugmentConfig(keep_image_prob=keep_image_prob)
    augmenter = TaskAugmenter(config)
    return augmenter.augment_task(trajectory)