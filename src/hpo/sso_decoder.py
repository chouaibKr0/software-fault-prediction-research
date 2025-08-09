import numpy as np

_TRANSFORMATIONS = {
    "baseline",
    "smooth_bounded",
    "probabilistic",
    "deterministic_probabilistic",
    "symmetric_bounded",
    "differentiable",
    "floor",
    "modulo",
    "gaussian",
    "lerp",
}

# 1. Baseline
def _decode_position_baseline(self, pos):
    """
    Map optimizer's real-valued position to actual hyperparameter values to try.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(round(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(round(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            idx = int(round(pos[i]))
            idx = max(0, min(idx, len(info["choices"]) - 1))
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp
# 2. Smooth_bounded
def _decode_position_sigmoid(self, pos):
    """
    Map optimizer's real-valued position using sigmoid activation.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            sigmoid_val = 1 / (1 + np.exp(-pos[i]))
            scaled_val = sigmoid_val * (info.get("max_val", 1000))
            hp[info["name"]] = int(10 ** scaled_val)
        elif info["type"] == "int":
            sigmoid_val = 1 / (1 + np.exp(-pos[i]))
            scaled_val = sigmoid_val * (info.get("max_val", 100))
            hp[info["name"]] = int(scaled_val)
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            sigmoid_val = 1 / (1 + np.exp(-pos[i]))
            idx = int(sigmoid_val * len(info["choices"]))
            idx = max(0, min(idx, len(info["choices"]) - 1))
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 3. Probabilistic
def _decode_position_softmax(self, pos):
    """  
    Map optimizer's real-valued position using softmax for categorical selection.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(round(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(round(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            # Create logits for each choice
            num_choices = len(info["choices"])
            logits = np.full(num_choices, pos[i])  # Use same value for all
            # Add small random perturbation to break ties
            logits += np.random.normal(0, 0.1, num_choices)
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Sample based on probabilities
            idx = np.random.choice(num_choices, p=probabilities)
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 4. Deterministic Probabilistic
def _decode_position_softmax_argmax(self, pos):
    """
    Map optimizer's real-valued position using softmax then argmax (deterministic).
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(round(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(round(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            # Create logits based on position value
            num_choices = len(info["choices"])
            logits = np.linspace(-pos[i], pos[i], num_choices)
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Take argmax (deterministic)
            idx = np.argmax(probabilities)
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 5. Symmetric Bounded
def _decode_position_tanh(self, pos):
    """
    Map optimizer's real-valued position using tanh activation.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            tanh_val = np.tanh(pos[i])  # Maps to [-1, 1]
            scaled_val = (tanh_val + 1) / 2  # Maps to [0, 1]
            scaled_val *= info.get("max_exp", 5)  # Scale for log space
            hp[info["name"]] = int(10 ** scaled_val)
        elif info["type"] == "int":
            tanh_val = np.tanh(pos[i])  # Maps to [-1, 1]
            scaled_val = (tanh_val + 1) / 2  # Maps to [0, 1]
            scaled_val *= info.get("max_val", 100)
            hp[info["name"]] = int(scaled_val)
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            tanh_val = np.tanh(pos[i])  # Maps to [-1, 1]
            scaled_val = (tanh_val + 1) / 2  # Maps to [0, 1]
            idx = int(scaled_val * len(info["choices"]))
            idx = max(0, min(idx, len(info["choices"]) - 1))
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 6. Differentiable
def _decode_position_gumbel_softmax(self, pos, temperature=1.0):
    """
    Map optimizer's real-valued position using Gumbel-Softmax for differentiable sampling.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(round(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(round(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            num_choices = len(info["choices"])
            
            # Create logits
            logits = np.full(num_choices, pos[i])
            
            # Add Gumbel noise
            gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, num_choices)))
            logits += gumbel_noise
            
            # Apply temperature scaling and softmax
            logits = logits / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Take argmax for hard assignment
            idx = np.argmax(probabilities)
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp


# 7. Floor/Ceiling (Alternative discretization)
def _decode_position_floor(self, pos):
    """
    Map optimizer's real-valued position using floor for discretization.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(np.floor(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(np.floor(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            idx = int(np.floor(pos[i]))
            idx = max(0, min(idx, len(info["choices"]) - 1))
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 8. Modulo-based (Periodic mapping)
def _decode_position_modulo(self, pos):
    """
    Map optimizer's real-valued position using modulo for periodic wrapping.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            # Use modulo in log space then convert
            log_val = float(pos[i]) % info.get("log_range", 5)  # Default log range
            hp[info["name"]] = int(10 ** log_val)
        elif info["type"] == "int":
            # Wrap around integer range
            int_range = info.get("max_val", 100)
            hp[info["name"]] = int(abs(pos[i])) % int_range
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            # Periodic wrapping around categorical choices
            idx = int(abs(pos[i])) % len(info["choices"])
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 9. Gaussian/Normal sampling
def _decode_position_gaussian(self, pos, noise_std=0.1):
    """
    Map optimizer's real-valued position adding Gaussian noise around rounded values.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            # Add Gaussian noise in log space
            noisy_val = float(pos[i]) + np.random.normal(0, noise_std)
            hp[info["name"]] = int(round(10 ** noisy_val))
        elif info["type"] == "int":
            # Add Gaussian noise then round
            noisy_val = pos[i] + np.random.normal(0, noise_std)
            hp[info["name"]] = int(round(noisy_val))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            # Add Gaussian noise to position then round to index
            noisy_val = pos[i] + np.random.normal(0, noise_std)
            idx = int(round(noisy_val))
            idx = max(0, min(idx, len(info["choices"]) - 1))
            hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp

# 10. Linear interpolation (for ordered categories)
def _decode_position_lerp(self, pos):
    """
    Map optimizer's real-valued position using linear interpolation between adjacent choices.
    """
    hp = {}
    for i, info in enumerate(self.param_info):
        if info["type"] == "log":
            hp[info["name"]] = 10 ** float(pos[i])
        elif info["type"] == "int_log":
            hp[info["name"]] = int(round(10 ** float(pos[i])))
        elif info["type"] == "int":
            hp[info["name"]] = int(round(pos[i]))
        elif info["type"] == "linear":
            hp[info["name"]] = float(pos[i])
        elif info["type"] == "cat":
            # Linear interpolation between adjacent categorical choices
            num_choices = len(info["choices"])
            if num_choices == 1:
                hp[info["name"]] = info["choices"][0]
            else:
                # Map position to continuous index space [0, num_choices-1]
                continuous_idx = pos[i] % num_choices
                if continuous_idx < 0:
                    continuous_idx += num_choices
                
                # Get adjacent indices
                lower_idx = int(np.floor(continuous_idx))
                upper_idx = (lower_idx + 1) % num_choices
                
                # Interpolation weight
                weight = continuous_idx - lower_idx
                
                # For categorical, we can't truly interpolate, so probabilistically choose
                if np.random.random() < weight:
                    idx = upper_idx
                else:
                    idx = lower_idx
                    
                hp[info["name"]] = info["choices"][idx]
        elif info["type"] == "constant":
            hp[info["name"]] = info["choices"][0]
    return hp
    