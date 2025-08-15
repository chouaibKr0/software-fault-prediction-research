from typing import Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
import warnings
from ..utils import load_config
from typing import Optional
from pathlib import Path

DEFAULT_CONFIG_PATH: Optional[Path] = Path("config/evaluation/cross_validation_config.yaml")

def evaluate_model_cv(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    cv_config: Dict[str, Any] = load_config(DEFAULT_CONFIG_PATH),
    scoring: Union[str, list] = 'roc_auc',
) -> Dict[str, Any]:
    """
    Evaluate a model using cross-validation with config-specified splitting.
    
    Args:
        model: A scikit-learn estimator instance.
        X: Feature data, either NumPy array or pandas DataFrame.
        y: Target labels as a numpy array.
        cv_config: Dict of cross-validation settings, e.g.:
            {
                'method': 'stratified_kfold',
                'n_splits': 5,
                'shuffle': True,
                'random_state': 42
            }
        scoring: Scoring metric name or list of metric names (compatible with sklearn).
    
    Returns:
        Dictionary with mean and fold-by-fold scores, e.g.:
            {
                'mean_roc_auc': 0.85,
                'fold_roc_auc_scores': [0.82, 0.87, 0.85, 0.84, 0.86]
            }
    """
    # Fixed: Direct access to cv_config instead of nested lookup
    method = cv_config.get('method', 'stratified_kfold')
    n_splits = cv_config.get('n_splits', 5)
    shuffle = cv_config.get('shuffle', True)
    random_state = cv_config.get('random_state', 42)
    
    # Currently supports only stratified k-fold for classification
    if method.lower() != 'stratified_kfold':
        raise NotImplementedError(f"Cross-validation method '{method}' not supported.")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Support multiple or single scoring metric(s)
    if isinstance(scoring, str):
        scoring = [scoring]
    
    fold_scores = {score_name: [] for score_name in scoring}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Fixed: Handle both DataFrame and numpy array indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = None
        
        # Try to get predicted probabilities for scorers that require it
        if any(s in ['roc_auc', 'average_precision'] for s in scoring):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)
                # Handle binary and multiclass cases
                if y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_val)
            else:
                warnings.warn("Model does not support predict_proba or decision_function. Some metrics may be invalid.")
        
        for score_name in scoring:
            scorer = get_scorer(score_name)
            
            if score_name in ['roc_auc', 'average_precision']:
                if y_proba is None:
                    raise ValueError(f"Scorer '{score_name}' requires probability estimates, but model does not provide them.")
                # Fixed: Use scorer directly instead of private _score_func
                score = scorer(model, X_val, y_val)
            else:
                score = scorer(model, X_val, y_val)
            
            fold_scores[score_name].append(score)
    
    # Fixed: Corrected dictionary comprehension syntax
    results = {
        f'mean_{k}': np.mean(v) for k, v in fold_scores.items()
    }
    
    results.update({
        f'fold_{k}_scores': v for k, v in fold_scores.items()
    })
    
    return results

def evaluate_model_cv_mean(
    model: BaseEstimator,
    X,
    y,
    cv_config: Dict[str, Any]= load_config(DEFAULT_CONFIG_PATH),
    scoring: Union[str, list] = 'roc_auc'
) -> Dict[str, float]:
    """
    Wrapper around `evaluate_model_cv` returning a dictionary with formatted metric names and mean scores.
    
    Args:
        model: A scikit-learn estimator instance.
        X: Feature data.
        y: Target labels.
        cv_config: Cross-validation configuration dictionary.
        scoring: Single or list of scoring metric names.
    
    Returns:
        Dictionary with keys as formatted metric names and values as the mean scores, e.g.:
        {"roc_auc": 0.8} or {"roc_auc": 0.8, "accuracy": 0.9}
    """
    results = evaluate_model_cv(model, X, y, cv_config, scoring)
    
    if isinstance(scoring, str):
        return {scoring: results[f'mean_{scoring}']}
    else:
        return {
            metric: results[f'mean_{metric}']
            for metric in scoring
        }
