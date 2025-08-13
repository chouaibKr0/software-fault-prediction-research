
from .utils import load_config, setup_experiment
from .hpo.base_optimizer import BaseOptimizer, BaseModel

class ExperimentPipeline:
    def __init__(self, dataset_name: str, model_class: BaseModel, hpo_class: BaseOptimizer):
        self.dataset_name = dataset_name
        self.model_class = model_class
        self.hpo_class = hpo_class
        #setup exp
        ...


 




    def run(self):
        # setup
        data_loading_config= load_config("config/data/loading_config.yaml")
        data_preprocessoing_config = load_config("config/data/preprocessing_config.yaml")
        
        model_config = load_config("config/model/svm_config.yaml")
        multi_scoring = load_config("config/evaluation/evaluation_metrics_config.yaml").get("multi_metrics")
        single_scoring = load_config("config/evaluation/evaluation_metrics_config.yaml").get("single_metric")
        cv_config = load_config("config/evaluation/cross_validation_config.yaml")
        hpo_config = load_config("config/hpo/sso_config.yaml")    



        #load_and_preprocess_data
        #run_hpo
        #train_and_eval
        #store_experiment_results(self)

        ...
