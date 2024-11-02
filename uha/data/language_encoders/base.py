from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Dict, Any
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for language encoders"""
    max_length: int = 77
    padding: str = "max_length"
    truncation: bool = True
    device: str = "cpu"
    cache_size: int = 10000
    return_tensors: str = "pt"


class BaseLanguageEncoder(nn.Module, ABC):
    """Abstract base class for language encoders"""
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        self.config = config or EncoderConfig()
        self._cache: Dict[str, torch.Tensor] = {}
        
    @abstractmethod
    def encode(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Encode text into embeddings"""
        pass
    
    def forward(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Forward pass with caching"""
        if isinstance(text, str):
            if text in self._cache:
                return self._cache[text]
            
            result = self.encode(text)
            if len(self._cache) < self.config.cache_size:
                self._cache[text] = result
            return result
            
        return self.encode(text)
    
    def clear_cache(self) -> None:
        """Clear the encoding cache"""
        self._cache.clear()

    def to(self, device: torch.device) -> 'BaseLanguageEncoder':
        """Move encoder to device"""
        super().to(device)
        self.config.device = str(device)
        return self


# uha/data/language_encoders/clip_tokens.py

from transformers import CLIPTextModel, CLIPTokenizer
import torch

from .base import BaseLanguageEncoder, EncoderConfig


@dataclass
class CLIPEncoderConfig(EncoderConfig):
    """Configuration for CLIP encoder"""
    model_name: str = "openai/clip-vit-base-patch32"
    freeze_backbone: bool = True


class CLIPEncoder(BaseLanguageEncoder):
    """CLIP text encoder with improved caching and error handling"""
    
    def __init__(self, config: Optional[CLIPEncoderConfig] = None):
        super().__init__(config or CLIPEncoderConfig())
        self.config = config or CLIPEncoderConfig()
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name)
            self.model = CLIPTextModel.from_pretrained(self.config.model_name)
            
            if self.config.freeze_backbone:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            self.to(torch.device(self.config.device))
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP encoder: {str(e)}")
            raise

    def encode(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Encode text using CLIP"""
        try:
            if isinstance(text, str):
                text = [text]
                
            inputs = self.tokenizer(
                text,
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                return_tensors=self.config.return_tensors
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            return outputs.pooler_output
            
        except Exception as e:
            logger.error(f"Error encoding text with CLIP: {str(e)}")
            raise


# uha/data/language_encoders/no_encoder.py

import torch

from .base import BaseLanguageEncoder, EncoderConfig


class NoEncoder(BaseLanguageEncoder):
    """Pass-through encoder for cases where no encoding is needed"""
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__(config or EncoderConfig())
    
    def encode(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Return empty tensor for no encoding"""
        return torch.empty(0)


# uha/data/language_encoders/muse.py

import tensorflow_hub as hub
import tensorflow_text  # Required for MUSE
import torch
import numpy as np

from .base import BaseLanguageEncoder, EncoderConfig


@dataclass
class MUSEConfig(EncoderConfig):
    """Configuration for MUSE encoder"""
    model_url: str = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    output_size: int = 512


class MUSEEncoder(BaseLanguageEncoder):
    """MUSE text encoder with improved error handling and caching"""
    
    def __init__(self, config: Optional[MUSEConfig] = None):
        super().__init__(config or MUSEConfig())
        self.config = config or MUSEConfig()
        
        try:
            self.model = hub.load(self.config.model_url)
        except Exception as e:
            logger.error(f"Failed to initialize MUSE encoder: {str(e)}")
            raise

    def encode(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Encode text using MUSE"""
        try:
            if isinstance(text, str):
                text = [text]
                
            # MUSE requires CPU
            with torch.device('cpu'):
                embeddings = self.model(text).numpy()
                return torch.from_tensor(embeddings).to(self.config.device)
                
        except Exception as e:
            logger.error(f"Error encoding text with MUSE: {str(e)}")
            raise


# uha/data/language_encoders/factory.py

from typing import Dict, Type

from .base import BaseLanguageEncoder, EncoderConfig
from .clip_tokens import CLIPEncoder, CLIPEncoderConfig
from .muse import MUSEEncoder, MUSEConfig
from .no_encoder import NoEncoder


class EncoderFactory:
    """Factory for creating language encoders"""
    
    _encoders: Dict[str, Type[BaseLanguageEncoder]] = {
        'clip': CLIPEncoder,
        'muse': MUSEEncoder,
        'none': NoEncoder
    }
    
    _configs: Dict[str, Type[EncoderConfig]] = {
        'clip': CLIPEncoderConfig,
        'muse': MUSEConfig,
        'none': EncoderConfig
    }
    
    @classmethod
    def create(cls, 
               encoder_type: str,
               config: Optional[Dict[str, Any]] = None) -> BaseLanguageEncoder:
        """Create language encoder instance"""
        if encoder_type not in cls._encoders:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        encoder_cls = cls._encoders[encoder_type]
        config_cls = cls._configs[encoder_type]
        
        if config:
            config = config_cls(**config)
        else:
            config = config_cls()
            
        return encoder_cls(config)