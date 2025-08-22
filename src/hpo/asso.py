from .base_optimizer import BaseOptimizer
import math
import numpy as np
import copy
from . import sso_decoder
from ..evaluation.cross_validation import evaluate_model_cv_mean
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import logging

class Salp(object):
    """Individual salp representation from ASSO algorithm."""
    
    def __init__(self):
        self.__X = []
        self.__fitness = float('inf')
    
    def get_X(self):
        return self.__X
    
    def set_X(self, X):
        self.__X = X
    
    def get_fitness(self):
        return self.__fitness
    
    def set_fitness(self, fitness):
        self.__fitness = fitness

class ASSO(BaseOptimizer):
    
    def __init__(self, config=None, model=None, logger: logging.Logger=None):
        self.DEFAULT_CONFIG_PATH = Path("config/hpo/sso_config.yaml")
        super().__init__(config, model, logger)
        
        # Configuration
        optimizer_config = self.config.get("optimizer_config", {})
        self.num_salps = optimizer_config.get("n_salps", 30)
        self.max_iter = optimizer_config.get("max_iter", 100)
        self.strategy = optimizer_config.get("strategy", "basic")
        self.transformation_function = optimizer_config.get("transformation_function", "baseline")
        
        # ASSO algorithm attributes
        self.salps = []
        self.F = None  # Best salp (food position)
        self.c1 = None
        self.iteration = 1
        self.dim = None
        self.lb = None
        self.ub = None
        self._best_params = None
        self.param_info = None
        self.minimise = True  # ASSO default
    
    def get_hpo_name(self):
        return 'sso'
    
    def _parse_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Parse search space into parameter info format."""
        param_info = []
        
        for k, v in search_space.items():
            if isinstance(v, dict):
                t = v.get("type", "")
                if t == "log_uniform":
                    min_val = float(v["min_value"])
                    max_val = float(v["max_value"])
                    param_info.append({
                        "name": k,
                        "type": "log",
                        "lb": math.log10(min_val),
                        "ub": math.log10(max_val)
                    })
                elif t == "uniform":
                    min_val = float(v["min_value"])
                    max_val = float(v["max_value"])
                    param_info.append({
                        "name": k,
                        "type": "linear",
                        "lb": min_val,
                        "ub": max_val
                    })
                elif t == "int_uniform" or t == "int_log_uniform":
                    is_log = "log" in t
                    min_val = int(v["min_value"])
                    max_val = int(v["max_value"])
                    if is_log:
                        param_info.append({
                            "name": k,
                            "type": "int_log",
                            "lb": math.log10(float(min_val)),
                            "ub": math.log10(float(max_val))
                        })
                    else:
                        param_info.append({
                            "name": k,
                            "type": "int",
                            "lb": min_val,
                            "ub": max_val
                        })
                elif t == "categorical":
                    param_info.append({
                        "name": k,
                        "type": "cat",
                        "lb": 0,
                        "ub": len(v["choices"]) - 1,
                        "choices": v["choices"]
                    })
            elif isinstance(v, list):
                param_info.append({
                    "name": k,
                    "type": "cat",
                    "lb": 0,
                    "ub": len(v) - 1,
                    "choices": v
                })
            else:
                param_info.append({
                    "name": k,
                    "type": "constant",
                    "lb": 0,
                    "ub": 0,
                    "choices": [v]
                })
        
        return param_info
    
    def _setup_bounds(self):
        """Setup bounds arrays."""
        if not self.param_info:
            raise ValueError("Parameter info not initialized.")
        
        self.dim = len(self.param_info)
        self.lb = np.array([p["lb"] for p in self.param_info])
        self.ub = np.array([p["ub"] for p in self.param_info])
    
    def _initialise_position(self):
        """Initialize random position within bounds (from ASSO)."""
        return np.random.uniform(0, 1, self.dim) * (self.ub - self.lb) + self.lb
    
    def _create_salps(self):
        """Create and initialize salp swarm (adapted from ASSO)."""
        self.salps = []
        
        for i in range(self.num_salps):
            s = Salp()
            s.set_X(self._initialise_position())
            self.salps.append(s)
        
        # Initial fitness evaluation will be done in optimize method
        self.F = Salp()  # Initialize best salp
    
    def _decode_position(self, pos):
        """Decode position to hyperparameter values."""
        if not hasattr(sso_decoder, '_TRANSFORMATIONS'):
            return self._decode_position_baseline(pos)
        
        if self.transformation_function not in sso_decoder._TRANSFORMATIONS:
            return self._decode_position_baseline(pos)
        
        decoder_map = {
            "baseline": sso_decoder._decode_position_baseline,
            "smooth_bounded": sso_decoder._decode_position_sigmoid,
            "probabilistic": sso_decoder._decode_position_softmax,
            "deterministic_probabilistic": sso_decoder._decode_position_softmax_argmax,
            "symmetric_bounded": sso_decoder._decode_position_tanh,
            "differentiable": sso_decoder._decode_position_gumbel_softmax,
            "floor": sso_decoder._decode_position_floor,
            "modulo": sso_decoder._decode_position_modulo,
            "gaussian": sso_decoder._decode_position_gaussian,
            "lerp": sso_decoder._decode_position_lerp
        }
        
        decoder_func = decoder_map.get(self.transformation_function)
        if decoder_func:
            return decoder_func(self, pos)
        else:
            return self._decode_position_baseline(pos)
    
    def _decode_position_baseline(self, pos):
        """Baseline decoder implementation."""
        if not self.param_info:
            return {}
        
        result = {}
        for i, param in enumerate(self.param_info):
            name = param["name"]
            param_type = param["type"]
            value = pos[i]
            
            if param_type == "linear":
                result[name] = value
            elif param_type == "log":
                result[name] = 10 ** value
            elif param_type == "int":
                result[name] = int(round(value))
            elif param_type == "int_log":
                result[name] = int(round(10 ** value))
            elif param_type == "cat":
                idx = int(round(np.clip(value, param["lb"], param["ub"])))
                result[name] = param["choices"][idx]
            elif param_type == "constant":
                result[name] = param["choices"][0]
        
        return result
    
    def _update_fitness(self, objective_function):
        """Update fitness for all salps."""
        for s in self.salps:
            try:
                params = self._decode_position(s.get_X())
                fitness = objective_function(**params)
                # Convert to minimization (ASSO assumes minimization)
                s.set_fitness(-fitness if not np.isnan(fitness) else float('inf'))
            except Exception as e:
                s.set_fitness(float('inf'))
    
    def _update_c1(self):
        """Update c1 coefficient (Eq. 3.2 from ASSO paper)."""
        self.c1 = 2 * math.exp(-((4 * self.iteration / self.max_iter) ** 2))
    
    def _update_leader(self, ind):
        """Update leader salp position (Eq. 3.1 from ASSO paper)."""
        c2 = np.random.uniform(0, 1, self.dim)
        c3 = np.random.uniform(0, 1, self.dim)
        
        F = copy.deepcopy(self.F.get_X())
        X = np.zeros(self.dim)
        
        for i in range(self.dim):
            if c3[i] < 0.5:
                X[i] = F[i] + self.c1 * ((self.ub[i] - self.lb[i]) * c2[i])
            else:
                X[i] = F[i] - self.c1 * ((self.ub[i] - self.lb[i]) * c2[i])
        
        self.salps[ind].set_X(X)
    
    def _update_salps(self):
        """Update all salp positions (from ASSO algorithm)."""
        for i in range(self.num_salps):
            # Eq. 3.1 - Leader update
            if i < self.num_salps / 2:
                self._update_leader(i)
            # Eq. 3.4 - Follower update
            else:
                X1 = copy.deepcopy(self.salps[i].get_X())
                X2 = copy.deepcopy(self.salps[i-1].get_X())
                X = 0.5 * (X1 + X2)
                self.salps[i].set_X(X)
    
    def _iterate(self, objective_function):
        """Single iteration of ASSO algorithm."""
        self._update_c1()
        self._update_salps()
        
        # Apply bounds constraints
        for s in self.salps:
            X = copy.deepcopy(s.get_X())
            for i in range(self.dim):
                X[i] = np.clip(X[i], self.lb[i], self.ub[i])
            s.set_X(X)
        
        # Update fitness
        self._update_fitness(objective_function)
        
        # Update best solution
        for s in self.salps:
            if self.minimise:
                if s.get_fitness() < self.F.get_fitness():
                    self.F = copy.deepcopy(s)
            else:
                if s.get_fitness() > self.F.get_fitness():
                    self.F = copy.deepcopy(s)
        
        self.iteration += 1
    def objective_function(self, config, objective_metric, X, y):
        """Create objective function for cross-validation evaluation."""
        def objective_fn(**param):
            try:
                model = self.model.set_params(**param).model
                scores = evaluate_model_cv_mean(
                    model, X, y, cv_config=config, scoring=objective_metric
                )
                return np.float64(scores.get(objective_metric, float('inf')))
            except Exception as e:
                self.logger.warning(f"Failed to evaluate parameters: {e}")
                return float('inf')
        
        return objective_fn    
    def optimize(self, objective_function) -> Tuple[Dict[str, Any], float]:
        """Main optimization method using ASSO algorithm."""
        self.logger.info("-" * 80)
        self.logger.info(f"Starting ASSO (Amended Salp Swarm Optimizer)")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Max iterations: {self.max_iter}, Number of salps: {self.num_salps}")
        self.logger.info(f"Transformation function: {self.transformation_function}")
        
        # Setup optimization
        search_space = self.get_search_space(self.model)
        self.param_info = self._parse_search_space(search_space)
        self._setup_bounds()
        self._create_salps()
        
        # Initialize iteration counter
        self.iteration = 1
        
        # Initial fitness evaluation
        self._update_fitness(objective_function)
        
        # Sort salps and set initial best
        if self.minimise:
            self.salps = sorted(self.salps, key=lambda x: x.get_fitness(), reverse=False)
        else:
            self.salps = sorted(self.salps, key=lambda x: x.get_fitness(), reverse=True)
        
        self.F = copy.deepcopy(self.salps[0])
        
        self.logger.info(f"Iteration {self.iteration}: best fitness {-self.F.get_fitness():.6f}")
        
        # Main optimization loop
        while self.iteration <= self.max_iter:
            self._iterate(objective_function)
            
            if self.iteration <= self.max_iter:
                self.logger.info(f"Iteration {self.iteration}: best fitness {-self.F.get_fitness():.6f}")
        
        # Prepare results
        self._best_params = self._decode_position(self.F.get_X())
        best_fitness = -self.F.get_fitness()  # Convert back to maximization
        
        self.logger.info(f"Process terminated: best fitness {best_fitness:.6f}")
        self.logger.info(f"Best parameters: {self._best_params}")
        self.logger.info("-" * 80)
        
        return self._best_params, best_fitness
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best parameters found."""
        return self._best_params
