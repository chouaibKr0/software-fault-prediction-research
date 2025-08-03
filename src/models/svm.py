from .base_model import BaseModel
from ..utils import set_random_seeds
from sklearn.svm import SVC

class SVM_Wrapper(BaseModel):
    def __init__(self, config, random_state= 42):
        self.model = None
        self.config = config
        self.random_state = random_state
    
    def _create_model(self, **params):
        # Merge config model parameters with any passed params
        model_params = self.config.get("model_config", {}).copy()
        model_params.update(params)
        model_params["random_state"] = self.random_state
        return SVC(**model_params)