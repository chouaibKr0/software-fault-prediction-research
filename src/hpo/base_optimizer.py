from abc import ABC, abstractmethod
from ..models.base_model import BaseModel 
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from ..utils import load_config

class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimizers.
    Subclasses must implement the main optimization workflow.
    """
    DEFAULT_CONFIG_PATH: Optional[Path] = None

    def __init__(self, config:Dict[str, Any] = None, model: BaseModel = None, logger:Optional[logging.Logger] = logging.getLogger('ml_experiment')):
        self.config = config
        self.model = model
        self.logger = logger
        if config == None and self.DEFAULT_CONFIG_PATH != None:
               self.config = load_config(self.DEFAULT_CONFIG_PATH)
            
    @abstractmethod
    def get_hpo_name(self) -> str:
        pass
    
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
        return search_space
    
    @abstractmethod
    def objective_function(self, ev_config, objective_metric, X, y):
        """
        Create objective function for model evaluation.
        
        This method should return a callable that evaluates model parameters
        and returns a numeric score for optimization.
        
        Args:
            config: Configuration object for evaluation (e.g., CV config)
            objective_metric (str): Name of the metric to optimize
            X: Training features
            y: Training targets
            
        Returns:
            callable: Function that takes **param and returns float score
                     Should return float('inf') for failed evaluations
                     
        Note:
            Child classes must implement their specific evaluation logic
            (e.g., cross-validation, holdout validation, etc.)
        """
        pass
    @abstractmethod
    def optimize(self, objective_function) -> Tuple[Dict[str, Any], float]:
        pass
