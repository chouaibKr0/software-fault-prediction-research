from pathlib import Path
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
import yaml
import numpy as np
import json
import random
from dataclasses import dataclass, asdict
from .utils import get_config_by_name, get_project_root

@dataclass
class ExperimentResult:
    """Structure for experiment results"""
    experiment_id: str
    dataset_name: str
    model_name: str
    hpo_name: str
    hpo_config_hash: str
    start_time: str
    end_time: str
    optimization_time_seconds: float
    best_params: Dict[str, Any]
    best_score: float
    evaluation_metrics: Dict[str, float]
    config: Dict[str, Any]
    
class MLExperiment:
    """Enhanced experiment management following ML best practices"""
    
    def __init__(self, dataset_name: str, model_name: str, hpo_name: str, 
                 config: Dict[str, Any], base_dir: str = "experiments"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hpo_name = hpo_name
        self.config = config
        self.base_dir = Path(base_dir)
        
        # Create experiment identifier
        self.hpo_config_hash = self._get_hpo_config_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{hpo_name}_{self.hpo_config_hash}_{model_name}_{dataset_name}_{self.timestamp}"
        
        # Initialize tracking variables
        self.start_time = None
        self.end_time = None
        self.optimization_time = 0.0
        self.best_params = {}
        self.best_score = None
        self.evaluation_metrics = {}
        
        # Setup experiment infrastructure
        self.directories = self._create_directories()
        self.logger = self._setup_logging()
        self.mlflow_enabled = config.get('use_mlflow', False)
        
        if self.mlflow_enabled:
            self._setup_mlflow()
    
    def _get_hpo_config_hash(self) -> str:
        """Create a short hash of important HPO configuration parameters"""
        import hashlib
        
        # Extract important HPO parameters
        hpo_config = get_config_by_name(self.hpo_name)
        important_params = hpo_config.get('optimizer_config', {})
        self.hpo_important_params = important_params
        param_str = json.dumps(important_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def _create_directories(self) -> Dict[str, Path]:
        """Create experiment directory structure"""
        exp_dir = get_project_root() / self.base_dir / self.experiment_id
        
        directories = {
            'experiment_dir': exp_dir,
            'logs': exp_dir / 'logs',
            'configs': exp_dir / 'configs', 
            'results': exp_dir / 'results',
            'models': exp_dir / 'models',
            'plots': exp_dir / 'plots'
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return directories
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment-specific logging"""
        logger = logging.getLogger(f'experiment_{self.experiment_id}')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.directories['logs'] / f"{self.experiment_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_mlflow(self):
        """Setup MLflow tracking if enabled"""
        try:
            import mlflow
            
            # Set tracking URI
            tracking_uri = self.config.get('mlflow_tracking_uri', 'mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            experiment_name = f"{self.hpo_name}_{self.hpo_config_hash}_{self.model_name}_{self.dataset_name}_experiments"
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass  # Experiment already exists
            
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow tracking enabled: {tracking_uri}")
            
        except ImportError:
            self.logger.warning("MLflow not installed, proceeding without MLflow tracking")
            self.mlflow_enabled = False
    
    def start(self) -> 'MLExperiment':
        """Start the experiment"""
        self.start_time = datetime.now()
        
        import shutil

        # Copy entire project config directory
        project_config_dir = get_project_root() / 'config'  # or wherever your project configs are
        destination_config_dir = self.directories['configs']

        if project_config_dir.exists():
            # Remove existing destination if it exists (since copytree can't overwrite)
            if destination_config_dir.exists():
                shutil.rmtree(destination_config_dir)
            
            # Copy entire directory tree
            shutil.copytree(project_config_dir, destination_config_dir)
        else:
            print(f"Warning: Project config directory {project_config_dir} not found")

        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING ML EXPERIMENT: {self.experiment_id}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"HPO Method: {self.hpo_name}")
        self.logger.info(f"Start Time: {self.start_time}")
        self.logger.info("=" * 80)
        

        if self.mlflow_enabled:
            import mlflow
            self.mlflow_run = mlflow.start_run(run_name=self.experiment_id)
            mlflow.log_params({
                'dataset': self.dataset_name,
                'model': self.model_name,
                'hpo_method': self.hpo_name,
                'experiment_id': self.experiment_id
            })
            
            # Log configuration parameters
    
            mlflow.log_params({f"hpo_{k}": v for k, v in self.hpo_important_params.items()})

            mlflow.log_params({f"model_{k}": v for k, v in get_config_by_name(self.model_name).items()})
        
        return self
    
    
    def finish(self, best_params: Dict[str, Any], best_score: float , 
               evaluation_metrics: Dict[str, float], 
               additional_artifacts: Optional[Dict[str, Any]] = None) -> ExperimentResult:
        """Finish the experiment and save all results"""
        
        self.end_time = datetime.now()
        self.optimization_time = (self.end_time - self.start_time).total_seconds()
        self.best_params = best_params
        self.best_score = best_score
        self.evaluation_metrics = evaluation_metrics
        
        # Create experiment result object
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            hpo_name=self.hpo_name,
            hpo_config_hash=self.hpo_config_hash,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            optimization_time_seconds=self.optimization_time,
            best_params=best_params,
            best_score=best_score,
            evaluation_metrics=evaluation_metrics,
            config=self.config
        )
        
        # Save results
        self._save_results(result, additional_artifacts)
        
        # Log completion
        self.logger.info("-" * 80)
        self.logger.info(f"EXPERIMENT COMPLETED: {self.experiment_id}")
        self.logger.info(f"Duration: {self.optimization_time:.2f} seconds")
        self.logger.info(f"Best Score: {best_score:.6f}")
        self.logger.info(f"Best Params: {best_params}")
        self.logger.info(f"Evaluation Metrics: {evaluation_metrics}")
        self.logger.info("=" * 80)
        
        if self.mlflow_enabled:
            import mlflow
            mlflow.log_metrics({
                'best_score': best_score,
                'optimization_time': self.optimization_time,
                **evaluation_metrics
            })
            mlflow.log_params(best_params)
            mlflow.end_run()
        
        return result
    
    def _save_results(self, result: ExperimentResult, additional_artifacts: Optional[Dict[str, Any]]):
        """Save experiment results to various formats"""
        
        # Save main result as JSON
        results_path = self.directories['results'] / 'experiment_results.json'
        with open(results_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save separate files for easy access
        best_params_path = self.directories['results'] / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(result.best_params, f, indent=2)
        
        metrics_path = self.directories['results'] / 'evaluation_metrics.json'  
        with open(metrics_path, 'w') as f:
            json.dump(result.evaluation_metrics, f, indent=2)
        
        # Save additional artifacts if provided
        if additional_artifacts:
            artifacts_path = self.directories['results'] / 'additional_artifacts.json'
            with open(artifacts_path, 'w') as f:
                json.dump(additional_artifacts, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {self.directories['results']}")


def setup_experiment(dataset_name: str, model_name: str, hpo_name: str, 
                     config: Dict[str,Any]= None) -> MLExperiment:
    """
    Setup a complete ML experiment
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model (e.g., 'svm', 'random_forest')
        hpo_name: Name of HPO method (e.g., 'optuna', 'grid_search')
        config_files: List of config files to merge
        
    Returns:
        Initialized MLExperiment object
    """
    # Load and merge configurations (using existing function)
    if config == None:
        config = get_config_by_name('base')
    # Set random seeds
    seed = config.get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    # Create and start experiment
    experiment = MLExperiment(dataset_name, model_name, hpo_name, config)
    return experiment.start()

# Utility functions for experiment analysis
def load_experiment_result(experiment_id: str, base_dir: str = "experiments") -> ExperimentResult:
    ...
    """Load experiment result from saved files"""
    results_path = Path(base_dir) / experiment_id / 'results' / 'experiment_results.json'
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return ExperimentResult(**data)

def compare_experiments(experiment_ids: List[str], base_dir: str = "experiments") -> List[ExperimentResult]:
    ...
    """Load and compare multiple experiments"""
    results = []
    for exp_id in experiment_ids:
        try:
            result = load_experiment_result(exp_id, base_dir)
            results.append(result)
        except FileNotFoundError:
            print(f"Warning: Experiment {exp_id} not found")
    
    return results
