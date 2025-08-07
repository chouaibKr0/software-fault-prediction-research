from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Dict, Any
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, config, random_state= 42):
        self.model = None
        self.config = config
        self.random_state = random_state

    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the sklearn model."""
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


