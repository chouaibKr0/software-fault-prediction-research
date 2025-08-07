from abc import ABC, abstractmethod
from ..models.base_model import BaseModel 
from typing import Dict, Any, Optional, Tuple
import logging

class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimizers.
    Subclasses must implement the main optimization workflow.
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def _parse_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Parse or validate the search space structure."""
        pass

    def get_search_space(self, model: BaseModel) -> Dict[str, Any]:
        """Extract and parse the hyperparameter search space for a given model."""
        model_config = model.config
        search_space = model_config.get("search_space", {})
        return self._parse_search_space(search_space)

    @abstractmethod
    def _evaluate(self, model: BaseModel , params: Dict[str, Any] ) -> float:
        """Train and evaluate the given model with specified hyperparameters."""
        pass

    @abstractmethod
    def optimize(self, model: BaseModel, logger:Optional[logging.Logger] = None) -> Tuple[Dict[str, Any], float]:
        pass