"""
ag_cube_cm.models.factory
==========================

Registry pattern for Crop Models.
"""

from typing import Type, Dict, Callable
import logging

from .base import CropModel
from ag_cube_cm.config.schemas import SimulationConfig

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Type[CropModel]] = {}

def register_model(name: str) -> Callable:
    """
    Decorator to register a CropModel implementation.
    
    Parameters
    ----------
    name : str
        The string identifier for the model (e.g., 'dssat', 'caf', 'simple_model').
    """
    def wrapper(model_class: Type[CropModel]) -> Type[CropModel]:
        if not issubclass(model_class, CropModel):
            raise ValueError(f"Class {model_class.__name__} must inherit from CropModel.")
        _MODEL_REGISTRY[name.lower()] = model_class
        logger.debug(f"Registered crop model: '{name.lower()}'")
        return model_class
    return wrapper

def model_factory(config: SimulationConfig) -> CropModel:
    """
    Instantiate and return the configured CropModel.
    
    Parameters
    ----------
    config : SimulationConfig
        The validated simulation configuration.
        
    Returns
    -------
    CropModel
        An instantiated crop model ready for spatial orchestration.
    """
    model_name = config.GENERAL_INFO.model.lower()
    
    if model_name not in _MODEL_REGISTRY:
        # Try to dynamically import the module if it's not registered yet
        try:
            import importlib
            importlib.import_module(f"ag_cube_cm.models.{model_name}.base")
        except ImportError as e:
            logger.warning(f"Could not auto-import module for '{model_name}': {e}")
            
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
        
    model_class = _MODEL_REGISTRY[model_name]
    logger.info(f"Instantiating model '{model_name}' via factory.")
    return model_class(config)
