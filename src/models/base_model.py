from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from ..utils import load_config
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for a ml model wrraper."""
    DEFAULT_CONFIG_PATH: Optional[Path] = None

    def __init__(self, config: Dict[str, Any] = None, random_state= 42):
        self.model = None
        self.config = config
        self.random_state = random_state
        if config == None and self.DEFAULT_CONFIG_PATH != None:
               self.config = load_config(self.DEFAULT_CONFIG_PATH)

    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the sklearn model."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass
    

        
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""

            
        self.model = self._create_model(**params)
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        if self.model is None:
            return {}
        return self.model.get_params()


