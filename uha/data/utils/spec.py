"""
Module specification and instantiation utilities.
Provides type-safe, serializable representation of functions or classes with arguments.
"""

from dataclasses import dataclass
from functools import partial
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """
    JSON-serializable representation of a function or class with arguments.
    Useful for specifying particular classes or functions in config files while
    maintaining serializability and command-line overridability.
    """
    module: str
    name: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        """Validate and initialize specification."""
        if self.kwargs is None:
            self.kwargs = {}
        self._validate_spec()

    def _validate_spec(self):
        """Validate specification components."""
        if not isinstance(self.module, str):
            raise ValueError(f"Module must be string, got {type(self.module)}")
        if not isinstance(self.name, str):
            raise ValueError(f"Name must be string, got {type(self.name)}")
        if not isinstance(self.args, (tuple, list)):
            raise ValueError(f"Args must be tuple or list, got {type(self.args)}")
        if not isinstance(self.kwargs, dict):
            raise ValueError(f"Kwargs must be dict, got {type(self.kwargs)}")

    @classmethod
    def create(cls, 
               callable_or_full_name: Union[str, Callable, Type], 
               *args, 
               **kwargs) -> 'ModuleSpec':
        """
        Create a module spec from a callable or import string.

        Args:
            callable_or_full_name: Either the callable object or a fully qualified
                import string (e.g. "module.submodule:Function")
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            ModuleSpec instance
        """
        try:
            if isinstance(callable_or_full_name, str):
                if callable_or_full_name.count(':') != 1:
                    raise ValueError(
                        "Import string must be in format 'module.submodule:name'"
                    )
                module, name = callable_or_full_name.split(':')
            else:
                module, name = cls._infer_full_name(callable_or_full_name)

            return cls(module=module, name=name, args=args, kwargs=kwargs)
        
        except Exception as e:
            logger.error(f"Error creating ModuleSpec: {str(e)}")
            raise

    @classmethod
    def from_dict(cls, spec_dict: Dict[str, Any]) -> 'ModuleSpec':
        """Create ModuleSpec from dictionary representation."""
        required_keys = {'module', 'name', 'args', 'kwargs'}
        if not set(spec_dict.keys()) == required_keys:
            raise ValueError(
                f"Spec dict must contain exactly keys {required_keys}, "
                f"got {set(spec_dict.keys())}"
            )
        return cls(**spec_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'module': self.module,
            'name': self.name,
            'args': self.args,
            'kwargs': self.kwargs
        }

    def save(self, path: Union[str, Path]):
        """Save specification to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModuleSpec':
        """Load specification from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No spec file found at {path}")
        with path.open('r') as f:
            return cls.from_dict(json.load(f))

    def instantiate(self, **override_kwargs) -> Callable:
        """
        Instantiate the specified callable with stored arguments.
        
        Args:
            **override_kwargs: Optional kwargs that override stored kwargs

        Returns:
            Partial function with stored/override arguments
        """
        try:
            cls = self._import_callable()
            kwargs = {**self.kwargs, **override_kwargs}
            return partial(cls, *self.args, **kwargs)
        
        except Exception as e:
            logger.error(
                f"Error instantiating {self.module}:{self.name}: {str(e)}"
            )
            raise

    def to_string(self) -> str:
        """Convert specification to string representation."""
        args_str = ", ".join(map(str, self.args))
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        separator = ", " if args_str and kwargs_str else ""
        return (
            f"{self.module}:{self.name}"
            f"({args_str}{separator}{kwargs_str})"
        )

    @staticmethod
    def _infer_full_name(obj: Union[Callable, Type]) -> Tuple[str, str]:
        """Infer module and name from callable object."""
        if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
            return obj.__module__, obj.__name__
        
        raise ValueError(
            f"Could not infer identifier for {obj}. "
            "Please use fully qualified import string instead "
            "(e.g., 'module.submodule:name')"
        )

    def _import_callable(self) -> Union[Callable, Type]:
        """Import the specified callable."""
        try:
            module = importlib.import_module(self.module)
            return getattr(module, self.name)
        except ImportError as e:
            raise ImportError(f"Could not import module {self.module}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Could not find name {self.name} in module {self.module}"
            ) from e


class ModuleSpecRegistry:
    """Registry for managing multiple module specifications."""
    
    def __init__(self):
        self._specs: Dict[str, ModuleSpec] = {}

    def register(self, name: str, spec: ModuleSpec):
        """Register a module specification."""
        if name in self._specs:
            logger.warning(f"Overwriting existing spec: {name}")
        self._specs[name] = spec

    def get(self, name: str) -> ModuleSpec:
        """Get a registered specification."""
        if name not in self._specs:
            raise KeyError(f"No spec registered with name: {name}")
        return self._specs[name]

    def instantiate(self, name: str, **override_kwargs) -> Callable:
        """Instantiate a registered specification."""
        return self.get(name).instantiate(**override_kwargs)

    def save_all(self, directory: Union[str, Path]):
        """Save all specifications to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, spec in self._specs.items():
            spec.save(directory / f"{name}.json")

    @classmethod
    def load_directory(cls, directory: Union[str, Path]) -> 'ModuleSpecRegistry':
        """Load all specifications from directory."""
        directory = Path(directory)
        registry = cls()
        for spec_file in directory.glob("*.json"):
            name = spec_file.stem
            registry.register(name, ModuleSpec.load(spec_file))
        return registry


# Create global registry instance
SPEC_REGISTRY = ModuleSpecRegistry()