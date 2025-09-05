#!/usr/bin/env python3
import argparse
import warnings

from src.pipeline import ExperimentPipeline
from src.models.svm import SVM_Wrapper
from src.hpo.salp_swarm_optimizer import SalpSwarmOptimizer
from src.hpo.asso import ASSO

warnings.filterwarnings('ignore')


# -----------------------------
# Lookup utilities
# -----------------------------
def get_model_class(model_name: str):
    """Map model name to model class."""
    model_mapping = {
        'svm': SVM_Wrapper,
    }
    name = model_name.lower()
    if name not in model_mapping:
        available = ', '.join(model_mapping.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available models: {available}")
    return model_mapping[name]


def get_hpo_class(hpo_name: str):
    """Map HPO name to HPO class."""
    hpo_mapping = {
        'sso': SalpSwarmOptimizer,
        'asso': ASSO,
    }
    name = hpo_name.lower()
    if name not in hpo_mapping:
        available = 'sso, asso'
        raise ValueError(f'HPO method "{hpo_name}" not supported. Available methods: {available}')
    return hpo_mapping[name]


def verify_dataset(dataset_name: str) -> bool:
    """Verify existence of dataset."""
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
        "xerces-1.2", "xerces-1.3", "xerces-1.4", "xerces-init",
    }


# -----------------------------
# Command handlers
# -----------------------------
def run_hpo(args: argparse.Namespace) -> None:
    """Dispatch for `hpo` subcommands (sso, asso)."""
    # args.hpo_cmd contains the chosen subcommand name
    hpo_name = args.hpo_cmd
    if not verify_dataset(args.dataset):
        raise ValueError(f"Dataset '{args.dataset}' does not exist")

    model_cls = get_model_class(args.model)
    hpo_cls = get_hpo_class(hpo_name)

    # Collect per-HPO kwargs from parsed args
    hpo_kwargs = {}
    # Shared flags

    if getattr(args, 'seed', None) is not None:
        hpo_kwargs['seed'] = args.seed
    # SSO-specific
    if hpo_name == 'sso':
        if getattr(args, 'max_iter', None) is not None:
            hpo_kwargs['max_iter'] = args.max_iter
        if getattr(args, 'n_salps', None) is not None:
            hpo_kwargs['n_salps'] = args.n_salps
        if getattr(args, 'strategy', None) is not None:
            hpo_kwargs['strategy'] = args.strategy
        if getattr(args, 'tf', None) is not None:
            hpo_kwargs['tf'] = args.tf

    # ASSO-specific
    if hpo_name == 'asso' and getattr(args, 'alpha', None) is not None:
        if getattr(args, 'max_iter', None) is not None:
            hpo_kwargs['max_iter'] = args.max_iter
        if getattr(args, 'n_salps', None) is not None:
            hpo_kwargs['n_salps'] = args.n_salps
        if getattr(args, 'strategy', None) is not None:
            hpo_kwargs['strategy'] = args.strategy
        if getattr(args, 'tf', None) is not None:
            hpo_kwargs['tf'] = args.tf
    pipeline = ExperimentPipeline(args.dataset, model_cls, hpo_cls, hpo_kwargs)
    pipeline.run()


# -----------------------------
# Parser setup
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run ML experiments with datasets, models, and HPO methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HPO via SSO with per-strategy flags
  python main.py --dataset ant-1.3 --model svm hpo sso --max_iter 100 --n_salps 42 --strategy 40

  # HPO via ASSO with different knobs
  python main.py --dataset ant-1.3 --model svm hpo asso --max_iter 60 
""",
    )

    # Global options (apply to all commands)
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset to use',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model to use (e.g., svm)',
    )

    # Top-level subcommands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Commands')  # Python 3.7+ [2]
    # hpo command
    hpo_parser = subparsers.add_parser('hpo', help='Hyperparameter optimization')
    hpo_sub = hpo_parser.add_subparsers(dest='hpo_cmd', required=True, help='HPO methods')  # [2]

    # sso subcommand
    sso = hpo_sub.add_parser('sso', help='Salp Swarm Optimizer')
    sso.add_argument('--max_iter', type=int, help='Number of iterations')
    sso.add_argument('--n_salps', type=int, help='Number of salps')
    sso.add_argument('--strategy', type=str, help='Strategy')
    sso.add_argument('--tf', type=str, help='Transformation function')

    sso.set_defaults(func=run_hpo)  # dispatch pattern [8][2]

    # asso subcommand
    asso = hpo_sub.add_parser('asso', help='ASSO optimizer')
    asso.add_argument('--max_iter', type=int, help='Number of iterations')
    asso.add_argument('--n_salps', type=int, help='Number of salps')
    asso.add_argument('--strategy', type=str, help='Strategy')
    asso.add_argument('--tf', type=str, help='Transformation function')
    asso.set_defaults(func=run_hpo)  # [8][2]

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Dispatch to the chosen subcommand handler
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
    


if __name__ == "__main__":
    main()
