from .base_model import BaseModel
from ..utils import set_random_seeds
from sklearn.svm import SVC

class SVM_Wrapper(BaseModel):
    def __init__(self, config, random_state= 42):
        self.model = None
        self.config = config
        self.random_state = random_state
    
    def _create_model(self, **params):
        model_param = params
        model_param.update(self.config)
        model_param["random_state"]= 42
        return SVC(**model_param)