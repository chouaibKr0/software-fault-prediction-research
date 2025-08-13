from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.utils import load_config
from src.models.svm import SVM_Wrapper
from src.evaluation.cross_validation import evaluate_model_cv, evaluate_model_cv_mean
from src.hpo.salp_swarm_optimizer import SalpSwarmOptimizer

class Test:
    def __init__(self, dataset_name: str, model_name: str, hpo_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hpo_name = hpo_name
        self.data_loading_config= load_config("config/data/loading_config.yaml")
        self.data_preprocessoing_config = load_config("config/data/preprocessing_config.yaml")
        self.model_config = load_config("config/model/svm_config.yaml")
        self.multi_scoring = load_config("config/evaluation/evaluation_metrics_config.yaml").get("multi_metrics")
        self.single_scoring = load_config("config/evaluation/evaluation_metrics_config.yaml").get("single_metric")
        self.cv_config = load_config("config/evaluation/cross_validation_config.yaml")
        self.hpo_config = load_config("config/hpo/sso_config.yaml")

    def test_load_and_preprocess_data(self):
        self.data_loader = DatasetLoader(self.data_loading_config)
        #print(f"dataset dir: {load_config("config/data/loading_config.yaml").get("dataset_dir", "")}")
        df = self.data_loader.load_dataset(f"{self.dataset_name}.csv")
        print("raw dataset")
        print(df.head())
        self.data_preprocessor = DataPreprocessor(self.data_preprocessoing_config) 
        #print(f"dataset dir: {load_config("config/data/preprocessing_config.yaml")}")

        df = self.data_preprocessor.handle_missing_values(df)
        print("dataset with no missval")
        print(df.head())
        X, y = self.data_preprocessor.separate_features_and_target(df)
        print("X")
        print(X.head())
        print("y")
        print(y.head())
        X = self.data_preprocessor.select_features(X)
        print("selecteded features")
        print(X.head())
        X = self.data_preprocessor.scale_features(X)
        print("scaled features")
        print(X.head())
        y = self.data_preprocessor.encode_label(y)
        print("label encoded")
        print(y)
        return X,y
    def test_model(self, X,y):
        svm = SVM_Wrapper(self.model_config)
        svm.set_params()
        print(svm.get_params())
        scoring = self.multi_scoring
        results = evaluate_model_cv(svm.model, X, y, self.cv_config, scoring)

        print(results)
        print(evaluate_model_cv_mean(svm.model, X, y, self.single_scoring))
    
    def test_sso(self, X,y):
        svm_hpo = SalpSwarmOptimizer(self.hpo_config, SVM_Wrapper(self.model_config))  
        cv_config= self.cv_config
        scoring = self.single_scoring
        results = svm_hpo.optimize(svm_hpo.objective_function(cv_config,scoring,X,y))
        print(results)


if __name__ == '__main__':
    mytest = Test('ant-1.3','','')
    X, y = mytest.test_load_and_preprocess_data()
    mytest.test_sso(X,y)
    #({'C': 0.5670795711425214, 'kernel': 'poly', 'gamma': 0.0745212371804279, 'degree': 5, 'coef0': -0.15607152066820035}, np.float64(-0.8761904761904763))