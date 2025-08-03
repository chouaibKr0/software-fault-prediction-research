from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Dict, Any
import pandas as pd
import numpy as np

class BaseModel(ABC):

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

    
    def fit(self, X:pd.DataFrame ,y:pd.Series):
        """Fit the model."""
        if self.model is None:
            self.set_params()
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        self.model.fit(X, y)
        return self

    def predict(self,X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    @abstractmethod
    def save(self, filepath):
        # Optional: implement save logic
        pass
    @abstractmethod
    def load(self, filepath):
        # Optional: implement load logic
        pass
