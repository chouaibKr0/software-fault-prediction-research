from .base_model import BaseModel
from ..utils import set_random_seeds
from sklearn.svm import SVC

class SVM_Wrapper(BaseModel):

    def _create_model(self, **params):
        model_param = params
        model_param.update(self.config.get("model_config",{}))
        model_param["random_state"]= 42
        return SVC(**model_param)