from .base_model import BaseModel
from ..utils import set_random_seeds
from sklearn.svm import SVC
from pathlib import Path
class SVM_Wrapper(BaseModel):
    def __init__(self, config=None, random_state=42):
        self.DEFAULT_CONFIG_PATH = Path("config/model/svm_config.yaml")
        super().__init__(config, random_state)
    def _create_model(self, **params):
        model_param = params
        model_param.update(self.config.get("model_config",{}))
        model_param["random_state"]= 42
        return SVC(**model_param)
    
    def get_model_name(self):
        return 'svm'