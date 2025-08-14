from functools import reduce
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Optional, Any
from pathlib import Path
from ..utils import load_config

class DataPreprocessor:
    """Class for preprocessing loaded dataset"""

    DEFAULT_CONFIG_PATH: Optional[Path] = Path("config/data/preprocessing_config.yaml")

    def __init__(self, config:Optional[Dict[str, Any]]=None):
        self.config = config
        if config == None and self.DEFAULT_CONFIG_PATH != None:
               self.config = load_config(self.DEFAULT_CONFIG_PATH)
                    
    
    def handle_missing_values(self, df: pd.DataFrame, categorical_strategy: str = 'drop') -> pd.DataFrame:
        """
        Handles missing values in the provided DataFrame using strategies specified in the config.

        Numeric columns are imputed using the strategy defined in the config (default: 'median').
        Categorical columns are imputed using the specified categorical_strategy (default: 'drop').

        Parameters:
            df (pd.DataFrame): Input DataFrame with possible missing values.
            categorical_strategy (str): Strategy for imputing categorical columns ('most_frequent', 'constant', etc.).

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        
        """
        strategy = self.config.get('missing_values',{}).get('strategy', 'median')
        if df.isnull().sum().sum() == 0:
            return df
            
        
        df_imputed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0 and df_imputed[numeric_cols].isnull().any().any():
            if 'numeric_imputer' not in self.fitted_transformers:
                self.fitted_transformers['numeric_imputer'] = SimpleImputer(strategy=strategy)
                df_imputed[numeric_cols] = self.fitted_transformers['numeric_imputer'].fit_transform(df_imputed[numeric_cols])
            else:
                df_imputed[numeric_cols] = self.fitted_transformers['numeric_imputer'].transform(df_imputed[numeric_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0 and df_imputed[categorical_cols].isnull().any().any():
            if 'categorical_imputer' not in self.fitted_transformers:
                self.fitted_transformers['categorical_imputer'] = SimpleImputer(strategy=categorical_strategy)
                df_imputed[categorical_cols] = self.fitted_transformers['categorical_imputer'].fit_transform(df_imputed[categorical_cols])
            else:
                df_imputed[categorical_cols] = self.fitted_transformers['categorical_imputer'].transform(df_imputed[categorical_cols])
        
        return df_imputed
        
    def separate_features_and_target(self, df: pd.DataFrame, target_column: Optional[str]= None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separates the features and target variable from the DataFrame.

        If target_column is not specified, assumes the last column is the target.
        Otherwise, uses the specified column as the target.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing features and target.
            target_column (Optional[str]): Name of the target column. If None, uses the last column.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple containing the features DataFrame (X) and target Series (y).
        """
        if target_column is None:
            # Assume last column is target
            X = df.iloc[:, :-1].copy()
            y = df.iloc[:, -1].copy()
            target_column = df.columns[-1]
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
        return X,y
    
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selects a subset of features from the input DataFrame based on the configuration.

        Parameters:
            X (pd.DataFrame): Input features DataFrame.
            n_components (int): Number of principal components to keep.

        Returns:
            pd.DataFrame: DataFrame containing only the selected features.
        """
        selected_features = self.config.get("feature_selection", {}).get("selected_features", [
    "wmc", "max_cc", "loc", "cbo", "lcom", "rfc", "ca", "ce", "noc", "lcom3"])
        if not selected_features:
            return X
        
        # Select only specified features
        X_selected = X[selected_features]
        
        return X_selected
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales numeric features in the DataFrame using the scaler specified in the config.

        The config should have a 'scaling' section with a 'method' key, e.g.:
            {'scaling': {'method': 'standard'}}  # or 'minmax'

        Returns:
            pd.DataFrame: DataFrame with scaled numeric features.
        """

        method = self.config.get("scaling", {}).get("method", "standard")
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return    X_scaled

    def encode_label(self, y: pd.Series) -> pd.Series:
        """
        Encodes the target labels using label encoding.

        Parameters:
            y (pd.Series): Target variable to encode.

        Returns:
            pd.Series: Encoded target variable.
        """
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded
