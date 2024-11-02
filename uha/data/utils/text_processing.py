"""
Text processing utilities for language handling in the data pipeline.
Provides standardized text tokenization and embedding capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union, List
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import torch

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Base configuration for tokenizers."""
    max_length: int = 77
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "np"
    cache_size: int = 10000


@dataclass
class HFTokenizerConfig(TokenizerConfig):
    """Configuration for Hugging Face tokenizers."""
    model_name: str
    encode_with_model: bool = False


@dataclass
class CLIPConfig(TokenizerConfig):
    """Configuration for CLIP text processor."""
    model_name: str = "openai/clip-vit-base-patch32"


@dataclass
class MuseConfig(TokenizerConfig):
    """Configuration for MUSE embeddings."""
    model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    output_dim: int = 512


class BaseTextProcessor(ABC):
    """Abstract base class for text processors."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    def encode(self, strings: Union[str, Sequence[str]]) -> Any:
        """Encode text into tokens or embeddings."""
        pass

    def _maybe_cache_result(self, key: str, result: Any) -> Any:
        """Cache result if cache is not full."""
        if isinstance(key, str) and len(self._cache) < self.config.cache_size:
            self._cache[key] = result
        return result

    def clear_cache(self):
        """Clear the encoding cache."""
        self._cache.clear()


class HFTokenizer(BaseTextProcessor):
    """Hugging Face tokenizer wrapper with caching."""
    
    def __init__(self, config: HFTokenizerConfig):
        super().__init__(config)
        self.config = config
        
        try:
            from transformers import AutoTokenizer, FlaxAutoModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = None
            if config.encode_with_model:
                self.model = FlaxAutoModel.from_pretrained(config.model_name)
                
        except Exception as e:
            logger.error(f"Failed to initialize HF tokenizer: {str(e)}")
            raise

    def encode(self, strings: Union[str, Sequence[str]]) -> Dict[str, np.ndarray]:
        """Encode text using HF tokenizer."""
        try:
            # Check cache for single strings
            if isinstance(strings, str) and strings in self._cache:
                return self._cache[strings]
            
            inputs = self.tokenizer(
                strings,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors
            )
            
            if self.model is not None:
                outputs = self.model(**inputs)
                result = np.array(outputs.last_hidden_state)
            else:
                result = dict(inputs)
                
            return self._maybe_cache_result(strings, result)
            
        except Exception as e:
            logger.error(f"Error encoding text with HF tokenizer: {str(e)}")
            raise


class CLIPTextProcessor(BaseTextProcessor):
    """CLIP text processor with caching."""
    
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.config = config
        
        try:
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize CLIP processor: {str(e)}")
            raise

    def encode(self, strings: Union[str, Sequence[str]]) -> Dict[str, np.ndarray]:
        """Encode text using CLIP processor."""
        try:
            # Check cache for single strings
            if isinstance(strings, str) and strings in self._cache:
                return self._cache[strings]
            
            inputs = self.processor(
                text=strings,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors
            )
            
            # Add position IDs
            inputs["position_ids"] = np.expand_dims(
                np.arange(inputs["input_ids"].shape[1]), 
                axis=0
            ).repeat(inputs["input_ids"].shape[0], axis=0)
            
            return self._maybe_cache_result(strings, inputs)
            
        except Exception as e:
            logger.error(f"Error encoding text with CLIP processor: {str(e)}")
            raise


class MuseEmbedding(BaseTextProcessor):
    """MUSE text embedding with caching."""
    
    def __init__(self, config: MuseConfig):
        super().__init__(config)
        self.config = config
        
        try:
            import tensorflow_hub as hub
            import tensorflow_text  # Required for MUSE
            
            self.model = hub.load(config.model_path)
        except Exception as e:
            logger.error(f"Failed to initialize MUSE model: {str(e)}")
            raise

    def encode(self, strings: Union[str, Sequence[str]]) -> np.ndarray:
        """Encode text using MUSE model."""
        try:
            # Check cache for single strings
            if isinstance(strings, str) and strings in self._cache:
                return self._cache[strings]
            
            with tf.device("/cpu:0"):
                embeddings = self.model(strings).numpy()
            
            return self._maybe_cache_result(strings, embeddings)
            
        except Exception as e:
            logger.error(f"Error encoding text with MUSE: {str(e)}")
            raise


class TextProcessorFactory:
    """Factory for creating text processors."""
    
    @staticmethod
    def create_processor(processor_type: str, **config_kwargs) -> BaseTextProcessor:
        """Create text processor instance."""
        processors = {
            'hf': (HFTokenizer, HFTokenizerConfig),
            'clip': (CLIPTextProcessor, CLIPConfig),
            'muse': (MuseEmbedding, MuseConfig)
        }
        
        if processor_type not in processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
            
        processor_cls, config_cls = processors[processor_type]
        config = config_cls(**config_kwargs)
        
        return processor_cls(config)