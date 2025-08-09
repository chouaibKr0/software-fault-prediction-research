from .base_optimizer import BaseOptimizer
import math
import numpy as np
import random
import sso_decoder
from ..evaluation.cross_validation import evaluate_model_cv_mean

class SalpSwarmOptimizer(BaseOptimizer):

    def __init__(self, config, model = None, logger = ...):
        super().__init__(config, model, logger) 
        self.num_salps = self.config.get("optimizer_config",{}).get("n_salps",)
        self.max_iter = self.config.get("optimizer_config",{}).get("max_iter",)
        self.strategy = self.config.get("optimizer_config",{}).get("strategy",)
        self.transformation_function = self.config.get("optimizer_config",{}).get("transformation_function",)
 
    
    def _parse_search_space(self, search_space):
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
                    # integer (but will round in decode)
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
    
    def _decode_position(self, pos):
        """
        Map optimizer's real-valued position to actual hyperparameter values to try.
        """
        if self.transformation_function not in sso_decoder._TRANSFORMATIONS:
            raise ValueError(f"Unknown transformation function: {self.transformation_function}. Supported: {sso_decoder._TRANSFORMATIONS}")
        if self.transformation_function == "baseline":
            return sso_decoder._decode_position_baseline(self, pos)
        elif self.transformation_function == "smooth_bounded":
            return sso_decoder._decode_position_sigmoid(self, pos)
        elif self.transformation_function == "probabilistic":
            return sso_decoder._decode_position_softmax(self, pos)
        elif self.transformation_function == "deterministic_probabilistic":
            return sso_decoder._decode_position_softmax_argmax(self, pos)
        elif self.transformation_function == "symmetric_bounded":
            return sso_decoder._decode_position_tanh(self, pos)
        elif self.transformation_function == "differentiable":
            return sso_decoder._decode_position_gumbel_softmax(self, pos)
        elif self.transformation_function == "floor":
            return sso_decoder._decode_position_floor(self, pos)
        elif self.transformation_function == "modulo":
            return sso_decoder._decode_position_modulo(self, pos)
        elif self.transformation_function == "gaussian":
            return sso_decoder._decode_position_gaussian(self, pos)
        elif self.transformation_function == "lerp":
            return sso_decoder._decode_position_lerp(self, pos)

    

    
    def objective_function(self, config, objective_metric, X, y, **params):
        """Make an obj fn
            Arg:
            - config : cv config
            - objective_metric : single metric
            - X : feature space
            - y : label column
            - param : model parameters 
            Return: objective_fn()
        """
        def objective_fn():
            scores = evaluate_model_cv_mean(self.model, X, y, cv_config=config,scoring= objective_metric)
            return  scores.get(objective_metric,0)
    
        return objective_fn


    def optimize(self, objective_function):
        def _objective_fn(position):
            try:
                params = self._decode_position(position)
                score = objective_function()
                return score
            except Exception as e:
                print(f"[SSA Warning] Failed to evaluate params due to: {e}")
                return float("inf")

        
        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            for i in range(self.num_salps):
                if i == 0:
                    # Leader
                    self.positions[i] = self._update_leader(c1, t)
                else:
                    # Follower
                    self.positions[i] = self._update_follower(self.positions[i], self.positions[i-1])
                # Evaluate
                try:
                    fitness = _objective_fn(self.positions[i])
                except Exception:
                    fitness = float("inf")
                if fitness < self.food_fitness:
                    self.food_fitness = fitness
                    self.food_position = self.positions[i].copy()
            # Local search (optional)
            if self.strategy == "hybrid" and t >= self.max_iter / 2:
                self._brownian_local_search(_objective_fn)
        # Save best
        self._best_params = self._decode_position(self.food_position)
        return self._best_params

    def get_best_params(self):
        return self._best_params
    
    # Core SSA methods
    def levy_flight(self, beta=1.5):
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def _update_leader(self, c1, t):
        # Uses your code structure
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
                leader = self.food_position + 0.01 * self.levy_flight()
            else:
                for j in range(self.dim):
                    c2, c3 = np.random.rand(), np.random.rand()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    if c3 < 0.5:
                        leader[j] = self.food_position[j] + c1 * step
                    else:
                        leader[j] = self.food_position[j] - c1 * step
        elif self.strategy == "hybrid":
            if t < self.max_iter / 2 and random.random() < 0.3:
                new_pos = self.food_position + 0.01 * self.levy_flight()
            else:
                new_pos = np.zeros(self.dim)
                for j in range(self.dim):
                    c2, c3 = random.random(), random.random()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    if c3 < 0.5:
                        new_pos[j] = self.food_position[j] + c1 * step
                    else:
                        new_pos[j] = self.food_position[j] - c1 * step
                # Crossover
                partner_idx = random.randint(0, self.num_salps - 1)
                partner = self.positions[partner_idx]
                mask = np.random.rand(self.dim) < 0.5
                new_pos[mask] = partner[mask]
                leader = new_pos
        else:
            raise ValueError(f"Unknown SSA strategy: {self.strategy}")
        return np.clip(leader, self.lb, self.ub)

    def _update_follower(self, curr, prev):
        return (curr + prev) / 2

    def _brownian_local_search(self, _objective_fn):
        brownian = np.random.normal(0, 1, self.dim)
        candidate = self.food_position + brownian * (self.ub - self.lb) * 0.01
        candidate = np.clip(candidate, self.lb, self.ub)
        try:
            fit = _objective_fn(candidate)
            if fit < self.food_fitness:
                self.food_fitness = fit
                self.food_position = candidate.copy()
        except Exception:
            pass
