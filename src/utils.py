from pathlib import Path
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import yaml
import numpy as np
import json
import random


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent

def get_relative_path(path : str| Path) -> str | Path:
    return path

def setup_logging(log_level: str = "INFO", log_dir: str = "experiments/logs") -> logging.Logger:
    """
    Setup simple logging for experiments.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
    
    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ml_experiment')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter - simple and readable
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_project_root() / Path(log_dir) / f"experiment_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def load_config(config_relative_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file from the project root, regardless of current working directory.

    Args:
        config_relative_path: Path to YAML file, relative to the project root.

    Returns:
        Configuration dictionary.
    """
    project_root = get_project_root()  # This is a Path object
    config_path = project_root / config_relative_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    
    

def merge_configs(*config_paths: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and merge multiple config files in order.
    Later configs override earlier ones.
    
    Args:
        *config_paths: Paths to config files to merge
        
    Returns:
        Merged configuration dictionary
        
    Example:
        config = merge_configs(
            'config/base_config.yaml',
            'config/models/svm_config.yaml', 
            'config/experiments/comparison_config.yaml'
        )
    """
    merged_config = {}
    
    for config_path in config_paths:
        if config_path and Path(config_path).exists():
            config = load_config(config_path)
            merged_config.update(config)
            
    return merged_config

def load_base_config(base_config_path: str = None) -> Dict[str, Any]:
    """
    Load the base configuration file.
    
    Returns:
        Base configuration dictionary.
    """
    if base_config_path == None:
        base_config_path= 'config/base_config.yaml'
    return load_config(base_config_path)  # Adjust path as needed


def get_config_by_name(configuration_name: str) -> Dict[str, Any]:
    from .data.loader import DatasetLoader
    from .data.preprocessor import DataPreprocessor
    from .evaluation import cross_validation
    from .hpo.salp_swarm_optimizer import SalpSwarmOptimizer
    from .models.svm import SVM_Wrapper 
    config_name = configuration_name

    CLASS_REGISTRY = {
        'data_loader': DatasetLoader,
        'data_preprocessor': DataPreprocessor,
        'salp_swarm_optimizer': SalpSwarmOptimizer,
        'svm': SVM_Wrapper
    }

    # Optional: Add aliases for backward compatibility
    CLASS_ALIASES = {
        'loading': 'data_loader',
        'data_loading': 'data_loader',
        'preprocessing': 'data_preprocessor', 
        'data_preprocessing': 'data_preprocessor', 
        'sso': 'salp_swarm_optimizer',
        'ssa': 'salp_swarm_optimizer',
        'salp_swarm_algorithm':'salp_swarm_optimizer'
    }    
    def get_class_by_name(name: str):
        """
        Maps a name to a class from the CLASS_REGISTRY.
        
        Args:
            name (str): The name or alias of the class to retrieve
            
        Returns:
            class or None: The corresponding class if found, None otherwise
        """


        # First check if name is directly in the registry
        if name in CLASS_REGISTRY:
            return CLASS_REGISTRY[name]
        
        # Check if it's an alias and resolve it
        if name in CLASS_ALIASES:
            actual_key = CLASS_ALIASES[name]
            return CLASS_REGISTRY.get(actual_key)
        
        # Not found
        return None
    cls = get_class_by_name(config_name)
    if cls != None and cls().DEFAULT_CONFIG_PATH != None:
     return load_config(cls().DEFAULT_CONFIG_PATH)
    elif config_name == 'basic' or config_name == 'base':
        return load_base_config()
    elif config_name == 'cross_validation' or config_name == 'cv' or config_name == 'evaluation' or config_name == 'ev':
        if cross_validation.DEFAULT_CONFIG_PATH != None:
            return load_config(cross_validation.DEFAULT_CONFIG_PATH)
    else:
        raise ValueError(f"No configuration file found for '{configuration_name}'")



def set_random_seeds() -> None:
    """
    Set random seeds for reproducibility.
    """
    seed = load_base_config().get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    # Set sklearn random state if available
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass


def get_single_scoring(config_path: str | Path = None):
    if config_path == None:
        config_path = 'config/evaluation/evaluation_metrics_config.yaml'
    return load_config(config_path).get('single_metric', 'roc_auc')

def get_multi_scoring(config_path: str | Path = None):
    if config_path == None:
        config_path = 'config/evaluation/evaluation_metrics_config.yaml'
    return load_config(config_path).get('multi_metrics', {})