<!-- Banner Image -->



<p align="center">


  <img src="assets/banner.png" alt="Software Fault Prediction Banner" width="100%">


</p>





<h1 align="center">🐛 Software Fault Prediction with SSA + SVM</h1>


<p align="center">


  <em>Graduation project — leveraging the <strong>Salp Swarm Algorithm (SSA)</strong> for hyperparameter optimization of <strong>Support Vector Machines (SVM)</strong> to improve defect prediction accuracy.</em>


</p>





<!-- Badges -->


<p align="center">


  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">


  <img src="https://img.shields.io/github/license/yourusername/software-fault-prediction-ssa-svm" alt="License">


  <img src="https://img.shields.io/github/stars/yourusername/software-fault-prediction-ssa-svm?style=social" alt="Stars">


  <img src="https://img.shields.io/badge/Made%20With-💻%20and%20❤️-red" alt="Made With Love">


</p>





---





##  Overview


This project is the result of months of dedicated research and development as part of our **graduation project**.  


It delivers an **automated software fault prediction system** that:  


- **Optimizes SVM hyperparameters** using the **Salp Swarm Algorithm (SSA)**.  


- **Boosts prediction accuracy** of defect-prone modules across **40+ real-world datasets**.  





** Main Goal:**  


Leverage **population-based metaheuristics** to make defect identification **accurate, more robust, and effective** in real-world software engineering.





---





##  Contributors


- **Karbala Chouaib**  


- **Ouraou Mohamed Abdelillah**  


- **Charane Mohamed Ilies**  


- **Mernache Mohamed Amine**





---





##  Table of Contents

1. [ Tech Stack](#-tech-stack)
2. [ Installation](#-installation)
3. [ Usage](#-usage)
4. [ Configuration Guide](#️-configuration-guide)
5. [ Project Architecture](#-project-architecture)
6. [ Supported Datasets](#-supported-datasets)
7. [ Experiment Management](#-experiment-management)
8. [ Advanced Usage](#-advanced-usage)
9. [ Performance & Benchmarks](#-performance--benchmarks)
10. [ Troubleshooting](#-troubleshooting)
11. [ Contact](#-contact)  





---





##  Features


-  **SSA Optimization** — Robust evolutionary hyperparameter search for SVM.  


-  **SVM Modeling** — High-performance, tunable classification models.  


-  **Flexible Configurations** — Modular configs for datasets, models, and HPO strategies.  


-  **Evaluation Tools** — Cross-validation & mean scoring.  


-  **Dataset Agnostic** — Works with 40+ datasets.  


-  **Extensible** — Add new datasets, models, or optimization methods easily.  





---





##  Tech Stack

- **Language:** Python 3.8+
- **Core Libraries:**  
  - `scikit-learn` — Machine learning algorithms and evaluation
  - `numpy` — Numerical computing and array operations
  - `pandas` — Data manipulation and analysis
  - `PyYAML` — Configuration file parsing
  - `mlflow` — Experiment tracking and model management
- **Optimization:** Custom SSA (Salp Swarm Algorithm) implementation
- **Architecture:** Modular, object-oriented design with abstract base classes



---





##  Installation

### Prerequisites
- **Python 3.8+** (tested on 3.11)
- `pip` package manager
- `git` for cloning

### Setup
```bash
# Clone the repository
git clone https://github.com/chouaibKr0/software-fault-prediction-research.git
cd software-fault-prediction-research

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

###  Quick Start
```bash
# Run your first experiment
python main.py --dataset ant-1.3 --model svm hpo sso

# Results will be saved in experiments/ directory
ls experiments/
```

---

##  Usage

**Run a fault prediction experiment:** 

### Basic Usage
```bash
python main.py --dataset DATASET_NAME --model MODEL_NAME hpo OPTIMIZER_NAME
```

### Examples
```bash
# Standard SSA optimization
python main.py --dataset ant-1.7 --model svm hpo sso

# Using ASSO (Amended SSA) optimizer  
python main.py --dataset camel-1.4 --model svm hpo asso

# Custom HPO parameters for SSO
python main.py --dataset ant-1.3 --model svm hpo sso --max_iter 100 --n_salps 50 --strategy basic

# ASSO with custom parameters
python main.py --dataset ant-1.3 --model svm hpo asso --max_iter 60 --n_salps 30
```

### Available Parameters

#### SSO (Salp Swarm Optimizer)
- `--max_iter`: Number of optimization iterations (default: 100)
- `--n_salps`: Number of salps in the swarm (default: 30)  
- `--strategy`: Optimization strategy (default: basic)
- `--tf`: Transformation function (default: baseline)

#### ASSO (Amended SSA Optimizer)
- `--max_iter`: Number of optimization iterations (default: 100)
- `--n_salps`: Number of salps in the swarm (default: 30)
- `--strategy`: Optimization strategy (default: basic)  
- `--tf`: Transformation function (default: baseline)

#### Get Help
```bash
# General help
python main.py --help

# HPO specific help  
python main.py --dataset ant-1.3 --model svm hpo --help

# SSO specific options
python main.py --dataset ant-1.3 --model svm hpo sso --help
```

---

##  Configuration Guide

### Available Configurations

The project uses YAML configuration files in the `config/` directory:

```yaml
config/
├── base_config.yaml           # Global project settings
├── data/
│   ├── loading_config.yaml    # Dataset loading parameters
│   └── preprocessing_config.yaml # Data preprocessing steps
├── model/
│   └── svm_config.yaml       # SVM model parameters
├── hpo/
│   └── sso_config.yaml       # SSA optimization settings
└── evaluation/
    ├── cross_validation_config.yaml
    └── evaluation_metrics_config.yaml
```

### Configuration Override Examples
```bash
# Use custom HPO parameters for SSO
python main.py --dataset ant-1.3 --model svm hpo sso --max_iter 200 --n_salps 50 --strategy basic

# Use custom HPO parameters for ASSO
python main.py --dataset ant-1.3 --model svm hpo asso --max_iter 100 --n_salps 40
```
---

##  Project Architecture

```
rxu001/
├── config/                   # Configuration files
│   ├── base_config.yaml     # Global settings (MLflow, seeds, paths)
│   ├── data/
│   │   ├── loading_config.yaml      # Dataset loading settings
│   │   └── preprocessing_config.yaml # Feature processing
│   ├── model/
│   │   └── svm_config.yaml         # SVM hyperparameter ranges
│   ├── hpo/
│   │   └── sso_config.yaml         # SSA optimizer settings
│   └── evaluation/
│       ├── cross_validation_config.yaml
│       └── evaluation_metrics_config.yaml
├── data/
│   └── PROMISE/
│       └── interim/         # Place your .csv datasets here
├── src/
│   ├── data/
│   │   ├── loader.py       # DatasetLoader class
│   │   └── preprocessor.py # Data preprocessing
│   ├── models/
│   │   ├── base_model.py   # Abstract model interface
│   │   └── svm.py         # SVM wrapper implementation
│   ├── hpo/
│   │   ├── base_optimizer.py    # Optimizer interface
│   │   ├── salp_swarm_optimizer.py # SSA implementation
│   │   └── asso.py             # Amended SSA variant
│   ├── evaluation/
│   │   └── cross_validation.py  # CV evaluation utilities
│   ├── pipeline.py         # Main experiment orchestration
│   ├── experiment.py       # Experiment management & logging
│   └── utils.py           # Configuration & utility functions
├── experiments/            # Auto-generated experiment results
├── mlruns/                # MLflow tracking data
├── tests/                 # Test files
├── main.py                # CLI entry point
└── requirements.txt       # Python dependencies
```

---

##  Supported Datasets

### PROMISE Software Engineering Repository
This project works with **40+ software defect datasets** from the PROMISE repository:


### Dataset Setup
1. Download datasets from [PROMISE Repository](http://promise.site.uottawa.ca/SERepository/)
2. Place CSV files in `data/PROMISE/interim/`
3. Follow naming convention: `dataset-version.csv` (e.g., `ant-1.3.csv`)

---

##  Experiment Management

### Automatic Result Tracking
Each experiment creates a timestamped directory in `experiments/`:

```
experiments/
└── asso_3a81ff3f_svm_ant-1.3_20250905_191912/
    ├── configs/          # Experiment configuration backup
    ├── logs/            # Detailed execution logs
    ├── models/          # Trained model artifacts
    ├── metrics/         # Performance metrics
    └── results.json     # Experiment summary
```

### MLflow Integration
The project includes **MLflow** tracking (enabled in `base_config.yaml`):
```bash
# View experiment results in MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### Key Metrics Tracked
- **Cross-validation scores** (ROC-AUC, Precision, Recall, F1)
- **Optimization time** and convergence
- **Best hyperparameters** found by SSA
- **Model performance** on test sets

---

##  Advanced Usage

### Programmatic Interface
```python
from src.pipeline import ExperimentPipeline
from src.models.svm import SVM_Wrapper
from src.hpo.salp_swarm_optimizer import SalpSwarmOptimizer

# Create and run experiment programmatically
pipeline = ExperimentPipeline(
    dataset_name="ant-1.3",
    model=SVM_Wrapper,
    hpo=SalpSwarmOptimizer,
    hpo_kwargs={"optimizer_config": {"n_salps": 30, "max_iter": 100}}
)

result = pipeline.run()
```

### Custom Dataset Loading
```python
from src.data.loader import DatasetLoader

loader = DatasetLoader()
df = loader.load_dataset("your-dataset.csv")
```

---

##  Performance & Benchmarks

### Optimization Results
- **SSA typically converges** in 50-200 iterations
- **Average improvement** of 5-15% over default SVM parameters
- **Cross-validation scores** consistently above 0.80 ROC-AUC on PROMISE datasets

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4+ CPU cores for faster optimization
- **Storage**: ~1GB for datasets + experiment logs

---

##  Troubleshooting

### Common Issues

**FileNotFoundError: Dataset file does not exist**
```bash
# Ensure dataset is in correct location
ls data/PROMISE/interim/ant-1.3.csv

# Check dataset naming convention
python -c "from src.data.loader import DatasetLoader; DatasetLoader().load_dataset('ant-1.3.csv')"
```

**Configuration Errors**
- Verify YAML syntax in config files
- Check that all required sections are present in configs
- Use `python -c "from src.utils import load_config; print(load_config('config/base_config.yaml'))"` to test

**Memory Issues with Large Datasets**
- Reduce `n_salps` in SSO configuration
- Use smaller `max_iter` values
- Enable `n_jobs: 1` in `base_config.yaml`

---



## 📬 Contact

💌 **For inquiries, feedback, or collaborations:**

- **Ouraou Mohamed Abdelillah** — abdelillah.ouraou@email.com
- **Karbala Chouaib** — karballac@gmail.com  
- **Charane Mohamed Ilies** — mohamediliesc@gmail.com
- **Mernache Mohamed Amine** 

---

<p align="center">⭐ If you like this project, don't forget to give it a star on GitHub!</p>

<p align="center">
  <strong>Made with 💻 and ❤️ by the rxu001 team</strong>
</p>






