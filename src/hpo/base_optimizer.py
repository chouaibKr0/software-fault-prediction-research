from abc import ABC, abstractmethod
from ..models.base_model import BaseModel 
from typing import Dict, Any, Optional, Tuple
import logging


class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimizers.
    Subclasses must implement the main optimization workflow.
    """
    def __init__(self, config, model: BaseModel = None, logger:Optional[logging.Logger] = logging.getLogger('ml_experiment')):
        self.config = config
        self.model = model
        self.logger = logger
    
    def setModel(self, model: BaseModel)-> None:
        """Set model to optimize if it is not set"""
        if self.model != None:
            raise ValueError("Model has already been initialized")
        self.model = model

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
    def optimize(self, objective_function) -> Tuple[Dict[str, Any], float]:
        pass