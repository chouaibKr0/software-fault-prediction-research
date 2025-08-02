from functools import reduce
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def handle_missing_values(self, df: pd.DataFrame, categorical_strategy: str = 'drop') -> pd.DataFrame:
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
        

    def reduce_dimensionality(X: pd.DataFrame) -> pd.DataFrame:
        pass