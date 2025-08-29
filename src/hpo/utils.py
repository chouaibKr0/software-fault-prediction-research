from ..evaluation.cross_validation import evaluate_model_cv_mean
import numpy as np

def objective_function_cv_eval(self, config, objective_metric, X, y):
    """Create objective function for cross-validation evaluation."""
    def objective_fn(**param):
        try:
            model = self.model.set_params(**param).model
            scores = evaluate_model_cv_mean(
                model, X, y, cv_config=config, scoring=objective_metric
            )
            return np.float64(scores.get(objective_metric, -float('inf')))
        except Exception as e:
            self.logger.warning(f"Failed to evaluate parameters: {e}")
            return -float('inf')

    return objective_fn

import multiprocessing as mp

def _cv_worker(model_factory, param, X, y, config, objective_metric, out_q):
    try:
        # build a fresh model instance to avoid cross-process object issues
        model = model_factory().set_params(**param).model
        scores = evaluate_model_cv_mean(model, X, y, cv_config=config, scoring=objective_metric)
        out_q.put(np.float64(scores.get(objective_metric, -float('inf'))))
    except Exception as e:
        out_q.put(-float('inf'))

def objective_function_cv_eval_timeout(self, config, objective_metric, X, y, timeout_seconds=60):
    """Create objective function for cross-validation evaluation with a hard 60s timeout."""
    # model_factory should return a new wrapper/estimator like self.model's class each time
    # If self.model is lightweight and pickleable, one can capture the class and init params.
    ModelClass = type(self.model)
    init_kwargs = getattr(self.model, "init_kwargs", {}) if hasattr(self.model, "init_kwargs") else {}
    def model_factory():
        # Expect the wrapper to accept **init_kwargs**; adapt as needed for the project
        return ModelClass(**init_kwargs)

    def objective_fn(**param):
        out_q = mp.Queue(maxsize=1)
        p = mp.Process(target=_cv_worker, args=(model_factory, param, X, y, config, objective_metric, out_q))
        try:
            p.start()
            p.join(timeout_seconds)
            if p.is_alive():
                p.terminate()
                p.join()
                return -float('inf')
            # If finished, fetch result if available
            if not out_q.empty():
                return out_q.get()
            return -float('inf')
        except Exception as e:
            try:
                if p.is_alive():
                    p.terminate()
            finally:
                p.join()
            self.logger.warning(f"Failed to evaluate parameters: {e}")
            return -float('inf')

    return objective_fn
