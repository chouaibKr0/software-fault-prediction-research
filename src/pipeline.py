from .data.loader import DatasetLoader
from .utils import load_config


class ExperimentPipeline:
    def __init__(self, dataset_name: str, model_name: str, hpo_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hpo_name = hpo_name


    def load_and_preprocess_data(self):
        self.data_loader = DatasetLoader(load_config("config/data/loading_config.yaml"))
        df = self.data_loader.load_dataset(f"{self.dataset_name}.csv")
        # Apply any necessary preprocessing steps
        pass

    def build_model(self):
        # Build model based on self.model_name
        pass

    def run_hpo(self):
        # Implement HPO based on self.hpo_name over model
        pass

    def train_and_eval(self, model):
        # Train and evaluate model on HPO results; return metrics
        pass



    def run(self):
        pass
