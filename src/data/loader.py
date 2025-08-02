import pandas as pd
from ..utils import get_project_root

class DatasetLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_dir = config.get("dataset_dir", "data/PROMISE/interim")
    
    def load_dataset(self, dataset_file_name: str) -> pd.DataFrame:
        """Load dataset from the specified file name.
        Args:
            dataset_file_name (str): The name of the dataset file to load.
        Returns:
            dataset (pandas DataFrame): dataset as dataframe.
        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        full_path = f"{get_project_root()}/{self.dataset_dir}/{dataset_file_name}"

        if not pd.io.common.file_exists(full_path):
            raise FileNotFoundError(f"Dataset file {full_path} does not exist.")
        
        df = pd.read_csv(full_path)

        return df
