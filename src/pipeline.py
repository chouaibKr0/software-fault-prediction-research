
from .utils import  get_config_by_name, get_multi_scoring,get_single_scoring, load_config
from .experiment import setup_experiment
from .models.base_model import BaseModel
from .hpo.base_optimizer import BaseOptimizer
from .data.loader import DatasetLoader
from .data.preprocessor import DataPreprocessor
from .evaluation.cross_validation import evaluate_model_cv_mean
from src.hpo.utils import objective_function_cv_eval_timeout
from typing import Dict, Any


class ExperimentPipeline:
    def __init__(self, dataset_name: str, model: BaseModel, hpo: BaseOptimizer, hpo_kwargs: Dict[str, Any] = None ):
        self.dataset_name = dataset_name
        self.model = model
        self.hpo = hpo
        self.model_name = model().get_model_name()
        self.hpo_name = hpo().get_hpo_name()
        self.hpo_kwargs = hpo_kwargs



    def run(self):
        # Setup expriment
        experiment = setup_experiment(self.dataset_name, self.model_name, self.hpo_name)
        logger = experiment.logger
        # Load and process data
        dataLoader = DatasetLoader(load_config("config/data/loading_config.yaml"),logger)
        df = dataLoader.load_dataset(f'{self.dataset_name}.csv')
        dataPreprocessor = DataPreprocessor(get_config_by_name('preprocessing'), logger)
        df = dataPreprocessor.handle_missing_values(df)
        X, y =dataPreprocessor.separate_features_and_target(df)
        X = dataPreprocessor.select_features(X)
        X = dataPreprocessor.scale_features(X)
        y = dataPreprocessor.encode_label(y) 

        # Optimize
        model: BaseModel = self.model(get_config_by_name(self.model_name))
        cfg = get_config_by_name(self.hpo_name).copy()
        if self.hpo_kwargs is not None and len(self.hpo_kwargs) > 0:
            logger.warning(f'Overriding default HPO config with {self.hpo_kwargs}')
            cfg['optimizer_config'].update(self.hpo_kwargs)
        hpo:BaseOptimizer = self.hpo(cfg, model, logger)
        objective_function = objective_function_cv_eval_timeout(hpo, get_config_by_name('cv'), get_single_scoring(), X, y, 10)
        best_params, ng_score = hpo.optimize(objective_function)
        score = -ng_score
        # Evaluate Final Model
        evaluation = evaluate_model_cv_mean(model.set_params(**best_params).model, X, y, get_config_by_name('cv'), get_multi_scoring())
        # Save Experiment Results
        experiment.finish(best_params,score,evaluation)




