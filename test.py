from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.utils import load_config
from src.models.svm import SVM_Wrapper

class Test:
    def __init__(self, dataset_name: str, model_name: str, hpo_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hpo_name = hpo_name
    def test_load_and_preprocess_data(self):
        self.data_loader = DatasetLoader(load_config("config/data/loading_config.yaml"))
        #print(f"dataset dir: {load_config("config/data/loading_config.yaml").get("dataset_dir", "")}")
        df = self.data_loader.load_dataset(f"{self.dataset_name}.csv")
        print("raw dataset")
        print(df.head())
        self.data_preprocessor = DataPreprocessor(load_config("config/data/preprocessing_config.yaml")) 
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
    def test_model(self):
        svm = SVM_Wrapper(load_config("config/model/svm_config.yaml").get("model_config",{}))
        svm.set_params()
        print(svm.get_params())
        
    
if __name__ == '__main__':
    mytest = Test('ant-1.3','','')
    X, y = mytest.test_load_and_preprocess_data()
    mytest.test_model()