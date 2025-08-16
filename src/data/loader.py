import pandas as pd
from ..utils import get_project_root
from typing import Optional,Dict,Any
from pathlib import Path
from ..utils import load_config, get_relative_path
import logging
class DatasetLoader:
    """Class for Loading dataset"""
    DEFAULT_CONFIG_PATH: Optional[Path] = Path("config/data/loading_config.yaml")

    def __init__(self, config:Optional[Dict[str, Any]]=None, logger: logging.Logger=None):
        self.config = config
        self.dataset_dir = config.get("dataset_dir", "data/PROMISE/interim")
        if config == None and self.DEFAULT_CONFIG_PATH != None:
               self.config = load_config(self.DEFAULT_CONFIG_PATH)
        self.logger = logger
    
    def load_dataset(self, dataset_file_name: str) -> pd.DataFrame:
        """Load dataset from the specified file name.
        Args:
            dataset_file_name (str): The name of the dataset file to load.
        Returns:
            dataset (pandas DataFrame): dataset as dataframe.
        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        self.logger.info(f'Loading {dataset_file_name} dataset from {self.dataset_dir}')
        full_path = f"{get_project_root()}/{self.dataset_dir}/{dataset_file_name}"

        if not pd.io.common.file_exists(full_path):
            raise FileNotFoundError(f"Dataset file {full_path} does not exist.")
        
        df = pd.read_csv(full_path)
        self.logger.info("Dataset loaded succesufully")
        return df
