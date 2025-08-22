import argparse
import warnings
from src.pipeline import ExperimentPipeline
from src.models.svm import SVM_Wrapper 
from src.hpo.salp_swarm_optimizer import SalpSwarmOptimizer
from src.hpo.asso import ASSO
warnings.filterwarnings('ignore')

def get_model_class(model_name: str):
    """Map model name to model class."""
    model_mapping = {
        'svm': SVM_Wrapper
    }
    
    model_name = model_name.lower()
    if model_name not in model_mapping:
        available_models = ', '.join(model_mapping.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")
    
    return model_mapping[model_name]


def get_hpo_class(hpo_name: str):
    """Map HPO name to HPO class."""
    hpo_mapping = {
        'sso' : SalpSwarmOptimizer,
        'asso' :  ASSO
    }
    
    hpo_name = hpo_name.lower()
    if hpo_name not in hpo_mapping:
        available_hpo = 'sso, asso' #', '.join([k for k in hpo_mapping.keys() if k != 'no_hpo'])
        raise ValueError(f'HPO method "{hpo_name}" not supported. Available methods: {available_hpo}')
    
    return hpo_mapping[hpo_name]

def verify_dataset(dataset_name: str):
    """Verify existence of dataset"""
    return dataset_name.lower() in {
        "ant-1.3", "ant-1.4", "ant-1.5", "ant-1.6", "ant-1.7",
        "camel-1.0", "camel-1.2", "camel-1.4", "camel-1.6",
        "ivy-1.1", "ivy-1.4", "ivy-2.0",
        "jedit-3.2", "jedit-4.0", "jedit-4.1", "jedit-4.2", "jedit-4.3",
        "log4j-1.0", "log4j-1.1", "log4j-1.2",
        "lucene-2.0", "lucene-2.2", "lucene-2.4",
        "poi-1.5", "poi-2.0", "poi-2.5", "poi-3.0",
        "synapse-1.0", "synapse-1.1", "synapse-1.2",
        "tomcat-6.0",
        "velocity-1.4", "velocity-1.5", "velocity-1.6",
        "xalan-2.4", "xalan-2.5", "xalan-2.6", "xalan-2.7",
        "xerces-1.2", "xerces-1.3", "xerces-1.4", "xerces-init"
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run machine learning experiments with different datasets, models, and HPO methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
  python main.py --dataset ant-1.3 --model svm --hpo sso_basic_baseline
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Dataset to use (ant-1.3, ant-1.4, ant-1.5, ant-1.6, ant-1.7, camel-1.0, camel-1.2, camel-1.4, camel-1.6, ivy-1.1, ivy-1.4, ivy-2.0, jedit-3.2, jedit-4.0, jedit-4.1, jedit-4.2, jedit-4.3, log4j-1.0, log4j-1.1, log4j-1.2, lucene-2.0, lucene-2.2, lucene-2.4, poi-1.5, poi-2.0, poi-2.5, poi-3.0, synapse-1.0, synapse-1.1, synapse-1.2, tomcat-6.0, velocity-1.4, velocity-1.5, velocity-1.6, xalan-2.4, xalan-2.5, xalan-2.6, xalan-2.7, xerces-1.2, xerces-1.3, xerces-1.4, xerces-init)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Model to use (svm)'
    )
    
    parser.add_argument(
        '--hpo', 
        type=str, 
        required=True,
        help='Hyperparameter optimization method (sso)'
    )
    
    args = parser.parse_args()

    

    if not verify_dataset(args.dataset):
        raise ValueError(f"Dataset '{args.dataset}' does not exist")
    
    pipeline = ExperimentPipeline(args.dataset, get_model_class(args.model), get_hpo_class(args.hpo)) 
    pipeline.run()

    
    

if __name__ == "__main__":
    main()
