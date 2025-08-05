from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
import warnings

def evaluate_model_cv(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    cv_config: Dict[str, Any],
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
            'mean_score': 0.85,
            'fold_scores': [0.82, 0.87, 0.85, 0.84, 0.86]
        }
    """

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
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = None

        # Try to get predicted probabilities for scorers that require it
        if any(s in ['roc_auc', 'average_precision'] for s in scoring):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                # Some classifiers provide decision_function instead of predict_proba
                y_proba = model.decision_function(X_val)
            else:
                warnings.warn("Model does not support predict_proba or decision_function. Some metrics may be invalid.")

        for score_name in scoring:
            scorer = get_scorer(score_name)
            if score_name in ['roc_auc', 'average_precision']:
                if y_proba is None:
                    raise ValueError(f"Scorer '{score_name}' requires probability estimates, but model does not provide them.")
                score = scorer._score_func(y_val, y_proba)
            else:
                score = scorer._score_func(y_val, y_pred)
            fold_scores[score_name].append(score)

    # Compute mean score per metric
    results = {
        f'mean_{k}': np.mean(v) for k, v in fold_scores.items()
    }
    results.update({
        f'fold_{k}_scores': v for k, v in fold_scores.items()
    })

    return results





def evaluate_model_cv_mean(
    model, 
    X, 
    y, 
    cv_config: Dict[str, Any], 
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
        {"roc-auc": 0.8} or {"roc-auc": 0.8, "accuracy": 0.9}
    """
    results = evaluate_model_cv(model, X, y, cv_config, scoring)

    if isinstance(scoring, str):
        return {scoring: results[f'mean_{scoring}']}
    else:
        return {
            metric: results[f'mean_{metric}'] 
            for metric in scoring
        }
