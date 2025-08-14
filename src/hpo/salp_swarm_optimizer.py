from .base_optimizer import BaseOptimizer
import math
import numpy as np
import random
from . import sso_decoder
from ..evaluation.cross_validation import evaluate_model_cv_mean
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
class SalpSwarmOptimizer(BaseOptimizer):
    
    def __init__(self, config=None, model=None, logger=None):
        self.DEFAULT_CONFIG_PATH = Path("config/hpo/sso_config.yaml")
        super().__init__(config, model, logger)
        
        # Fixed initialization with proper defaults
        optimizer_config = self.config.get("optimizer_config", {})
        self.num_salps = optimizer_config.get("n_salps", 30)
        self.max_iter = optimizer_config.get("max_iter", 100)
        self.strategy = optimizer_config.get("strategy", "basic")
        self.transformation_function = optimizer_config.get("transformation_function", "baseline")
        
        # Initialize essential attributes
        self.positions = None
        self.food_position = None
        self.food_fitness = float('inf')
        self.dim = None
        self.lb = None
        self.ub = None
        self._best_params = None
        self.param_info = None
        


    def _parse_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the search space dict into a flat config list of
        dicts. Each item describes:
        - name: parameter name
        - type: (log, linear, int, cat)
        - lb, ub: lower, upper bounds
        - choices: for categorical
        """
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
                # List: treat as categorical
                param_info.append({
                    "name": k,
                    "type": "cat",
                    "lb": 0,
                    "ub": len(v) - 1,
                    "choices": v
                })
            else:
                # Single fixed value: degenerate interval
                param_info.append({
                    "name": k,
                    "type": "constant",
                    "lb": 0,
                    "ub": 0,
                    "choices": [v]
                })
        
        return param_info
    
    def _setup_bounds(self):
        """Setup lower and upper bounds arrays from parsed parameter info."""
        if not self.param_info:
            raise ValueError("Parameter info not initialized. Call _parse_search_space first.")
        
        self.dim = len(self.param_info)
        self.lb = np.array([p["lb"] for p in self.param_info])
        self.ub = np.array([p["ub"] for p in self.param_info])
        
    def _initialize_swarm(self):
        """Initialize the salp swarm positions randomly within bounds."""
        if self.dim is None or self.lb is None or self.ub is None:
            raise ValueError("Bounds not set. Call _setup_bounds first.")
        
        self.positions = np.random.uniform(
            self.lb, self.ub, (self.num_salps, self.dim)
        )
        self.food_fitness = float('inf')
        self.food_position = np.zeros(self.dim)
        
    def _setup_optimization(self, search_space: Dict[str, Any]):
        """Complete setup for optimization process."""
        self.param_info = self._parse_search_space(search_space)
        self._setup_bounds()
        self._initialize_swarm()
        
    def _decode_position(self, pos):
        """
        Map optimoptimizer_configizer's real-valued position to actual hyperparameter values to try.
        """
        if not hasattr(sso_decoder, '_TRANSFORMATIONS'):
            # Fallback if decoder module not available
            self.logger.warning("SSO decoder not available, using baseline decoding")
            return self._decode_position_baseline(pos)
        
        if self.transformation_function not in sso_decoder._TRANSFORMATIONS:
            raise ValueError(f"Unknown transformation function: {self.transformation_function}. "
                           f"Supported: {sso_decoder._TRANSFORMATIONS}")
        
        # Delegate to appropriate decoder function
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
        """Fallback baseline decoder if sso_decoder module unavailable."""
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
        """
        Main optimization method with proper interface compliance.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space definition
            
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        # Setup optimization
        search_space = self.get_search_space(self.model)
        self._setup_optimization(search_space)
        
        def _objective_fn(position):
            try:
                params = self._decode_position(position)
                score = objective_function(**params)
                return score if not np.isnan(score) else float('inf')
            except Exception as e:
                #self.logger.warning(f"Failed to evaluate position: {e}")
                return float('inf')
        
        # Initialize best solution by evaluating initial positions
        for i in range(self.num_salps):
            fitness = -_objective_fn(self.positions[i])
            if fitness < self.food_fitness:
                self.food_fitness = fitness
                self.food_position = self.positions[i].copy()
        
        # Main optimization loop
        for t in range(self.max_iter):
            # Calculate decreasing coefficient
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            
            for i in range(self.num_salps):
                if i == 0:
                    # Leader salp
                    self.positions[i] = self._update_leader(c1, t)
                else:
                    # Follower salp
                    self.positions[i] = self._update_follower(
                        self.positions[i], self.positions[i-1]
                    )
                
                # Ensure bounds
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # Evaluate fitness
                fitness = -_objective_fn(self.positions[i])
                
                # Update best solution
                if fitness < self.food_fitness:
                    self.food_fitness = fitness
                    self.food_position = self.positions[i].copy()
            
            # Optional local search for hybrid strategy
            if self.strategy == "hybrid" and t >= self.max_iter / 2:
                self._brownian_local_search(_objective_fn)
            
            # Log progress
            if self.logger and t % (self.max_iter // 10) == 0:
                self.logger.info(f"SSO Iteration {t}: Best fitness = {self.food_fitness:.6f}")
        
        # Save and return best solution
        self._best_params = self._decode_position(self.food_position)
        return self._best_params, self.food_fitness
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best parameters found."""
        return self._best_params
    
    # Core SSA algorithm methods
    
    def levy_flight(self, beta=1.5):
        """Generate Levy flight random walk."""
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        
        return u / (np.abs(v) ** (1 / beta))
    
    def _update_leader(self, c1, t):
        """Update leader salp position."""
        leader = np.empty(self.dim)
        
        if self.strategy == "basic":
            for j in range(self.dim):
                c2, c3 = np.random.rand(), np.random.rand()
                step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                
                if c3 < 0.5:
                    leader[j] = self.food_position[j] + c1 * step
                else:
                    leader[j] = self.food_position[j] - c1 * step
                    
        elif self.strategy == "levy_flights":
            if t < self.max_iter / 2 and np.random.rand() < 0.2:
                # Apply Levy flight
                levy_step = 0.01 * self.levy_flight()
                leader = self.food_position + levy_step
            else:
                # Standard update
                for j in range(self.dim):
                    c2, c3 = np.random.rand(), np.random.rand()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    
                    if c3 < 0.5:
                        leader[j] = self.food_position[j] + c1 * step
                    else:
                        leader[j] = self.food_position[j] - c1 * step
                        
        elif self.strategy == "hybrid":
            if t < self.max_iter / 2 and random.random() < 0.3:
                # Levy flight exploration
                levy_step = 0.01 * self.levy_flight()
                new_pos = self.food_position + levy_step
            else:
                # Standard position update
                new_pos = np.zeros(self.dim)
                for j in range(self.dim):
                    c2, c3 = random.random(), random.random()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    
                    if c3 < 0.5:
                        new_pos[j] = self.food_position[j] + c1 * step
                    else:
                        new_pos[j] = self.food_position[j] - c1 * step
            
            # Crossover with random salp
            if self.num_salps > 1:
                partner_idx = random.randint(0, self.num_salps - 1)
                partner = self.positions[partner_idx]
                mask = np.random.rand(self.dim) < 0.5
                new_pos[mask] = partner[mask]
            
            leader = new_pos
            
        else:
            raise ValueError(f"Unknown SSA strategy: {self.strategy}")
        
        return np.clip(leader, self.lb, self.ub)
    
    def _update_follower(self, curr, prev):
        """Update follower salp position with enhanced exploration."""
        # Standard SSA follower update with small random perturbation
        new_pos = (curr + prev) / 2
        
        # Add small random exploration
        noise = np.random.normal(0, 0.01, self.dim) * (self.ub - self.lb)
        new_pos += noise
        
        return np.clip(new_pos, self.lb, self.ub)
    
    def _brownian_local_search(self, _objective_fn):
        """Brownian motion-based local search around best solution."""
        try:
            # Generate Brownian motion step
            brownian = np.random.normal(0, 1, self.dim)
            step_size = 0.01 * (self.ub - self.lb)
            candidate = self.food_position + brownian * step_size
            candidate = np.clip(candidate, self.lb, self.ub)
            
            # Evaluate candidate
            fitness = -_objective_fn(candidate)
            
            # Update if better
            if fitness < self.food_fitness:
                self.food_fitness = fitness
                self.food_position = candidate.copy()
                
        except Exception as e:
            self.logger.warning(f"Brownian local search failed: {e}")
