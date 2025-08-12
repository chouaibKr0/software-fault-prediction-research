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





## 📖 Overview


This project is the result of months of dedicated research and development as part of our **graduation project**.  


It delivers an **automated software fault prediction system** that:  


- **Optimizes SVM hyperparameters** using the **Salp Swarm Algorithm (SSA)**.  


- **Boosts prediction accuracy** of defect-prone modules across **40+ real-world datasets**.  





**🎯 Main Goal:**  


Leverage **evolutionary optimization** to make defect identification **faster, more robust, and effective** in real-world software engineering.





---





## 👥 Contributors


- **Karbala Chouaib**  


- **Ouraou Mohamed Abdelillah**  


- **Charaane Mohamed Illies**  


- **Mohamed Amine XXXXX**





---





## 📌 Table of Contents


1. [✨ Features](#-features)  


2. [🛠 Tech Stack](#-tech-stack)  


3. [📥 Installation](#-installation)  


4. [🚀 Usage](#-usage)  


5. [⚙ Configuration](#-configuration)  


6. [📂 Project Architecture](#-project-architecture)  


7. [📊 Datasets](#-datasets)  


8. [🧪 Testing](#-testing)  


9. [📜 License](#-license)  


10. [📬 Contact](#-contact)  





---





## ✨ Features


- 🔍 **SSA Optimization** — Robust evolutionary hyperparameter search for SVM.  


- 📈 **SVM Modeling** — High-performance, tunable classification models.  


- 🛠 **Flexible Configurations** — Modular configs for datasets, models, and HPO strategies.  


- 📊 **Evaluation Tools** — Cross-validation & mean scoring.  


- 🌍 **Dataset Agnostic** — Works with 40+ datasets.  


- 📦 **Extensible** — Add new datasets, models, or optimization methods easily.  





---





## 🛠 Tech Stack


- **Language:** Python 3.x  


- **Core Libraries:**  


  - `numpy`  


  - `math`  


  - `random` (native)  


  - `typing`  


  - Custom modules:  


    - `src.hpo.base_optimizer`  


    - `src.hpo.sso_decoder`  


    - `src.evaluation.cross_validation`  





---





📥 Installation
Prerequisites

    Python 3.8+ (tested on 3.11)

    pip package manager

Setup

# Clone the repository
git clone https://github.com/yourusername/software-fault-prediction-ssa-svm.git

# Navigate to project directory
cd software-fault-prediction-ssa-svm

# Install dependencies
pip install -r requirements.txt

🚀 Usage

Run a fault prediction experiment:

python src/pipeline.py --config config/base_config.yaml

Workflow:

    Place datasets in the data/ folder.

    Set experiment configs in config/*.yaml.

    Run the pipeline with your chosen config file.

⚙ Configuration

Configuration files are located in the config/ directory:

    Data → config/data/

    Model → config/model/

    HPO (SSA) → config/hpo/

    Evaluation → config/evaluation/

Example:
base_config.yaml contains global experiment settings.
📂 Project Architecture

project_root/
├── config/
│   ├── data/
│   ├── evaluation/
│   ├── hpo/
│   ├── model/
│   └── base_config.yaml
├── data/                  # Datasets
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── hpo/
│   ├── models/
│   ├── pipeline.py
│   └── utils.py
├── tests/
│   ├── salp_swarm_optimizer.py
│   └── test.py
└── README.md

📊 Datasets

    Works with 40+ datasets for software defect prediction.

    Supported: PROMISE repository and CSV datasets.

    Place datasets inside data/ following naming conventions.

🧪 Testing

Run all tests:

python tests/test.py

Run SSA-specific tests:

python tests/salp_swarm_optimizer.py

📜 License

This project is licensed under the MIT License — see the LICENSE file for details.
📬 Contact

💌 For inquiries, feedback, or collaborations:

    Ouraou Mohamed Abdelillah — abdelillah.ouraou@email.com

    Karbala Chouaib — xxxxxxx@gmail.com

<p align="center">⭐ If you like this project, don't forget to give it a star on GitHub!</p>
