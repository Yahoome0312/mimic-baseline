"""
Utility helper functions
"""

import os
import json
import torch


class EarlyStopping:
    """Early stopping mechanism"""

    def __init__(self, patience=5, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Check if training should stop

        Args:
            score: Current score
            epoch: Current epoch

        Returns:
            True if score improved, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"Early stopping initialized: best_score = {score:.4f}")
            return True

        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[IMPROVED] Performance improved! New best_score = {score:.4f}")
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"[NO IMPROVE] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best epoch: {self.best_epoch}, Best score: {self.best_score:.4f}")

            return False


def set_seed(seed):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(gpu_id=None):
    """
    Get PyTorch device

    Args:
        gpu_id: GPU ID to use, None for auto-select

    Returns:
        torch.device
    """
    if gpu_id is not None:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_id}')
        else:
            print(f"Warning: CUDA not available, using CPU instead")
            return torch.device('cpu')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info():
    """Print device information"""
    print("\n" + "=" * 80)
    print("Device Information")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print("=" * 80 + "\n")


def load_class_config(dataset_name, class_names_dir=None, verbose=True):
    """
    Load complete class configuration from JSON file

    Args:
        dataset_name: Name of the dataset (e.g., 'mimic_cxr', 'chestxray14')
        class_names_dir: Directory containing class name JSON files
                        (default: project_root/class_names/)
        verbose: Print loading information (default: True)

    Returns:
        Dictionary with class configuration:
        {
            'class_names': List of class names,
            'num_classes': Number of classes,
            'task_type': Task type ('multi-label' or 'single-label'),
            'dataset_name': Dataset name
        }

    Raises:
        FileNotFoundError: If class names file doesn't exist
        ValueError: If JSON file is invalid
    """
    # Default to project root/class_names/
    if class_names_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        class_names_dir = os.path.join(project_root, 'class_names')

    # Construct file path
    config_file = os.path.join(class_names_dir, f'{dataset_name}.json')

    # Check if file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Class names configuration file not found: {config_file}\n"
            f"Available datasets: {list_available_datasets(class_names_dir)}"
        )

    # Load JSON
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_file}: {e}")

    # Extract class names
    if 'class_names' not in config:
        raise ValueError(f"'class_names' field not found in {config_file}")

    class_names = config['class_names']
    num_classes = len(class_names)
    dataset_display_name = config.get('dataset_name', dataset_name)

    # Get text prompt template and generate prompts
    template = config.get('text_prompt_template', 'There is {disease}.')
    text_prompts = [template.replace('{disease}', cls.lower().replace('_', ' ')) for cls in class_names]

    # Print info
    if verbose:
        print(f"\n{'='*80}")
        print(f"Loaded class configuration for: {dataset_display_name}")
        print(f"{'='*80}")
        print(f"Number of classes: {num_classes}")
        print(f"Task type: multi-label")
        print(f"Template: {template}")
        print(f"\nText prompts:")
        for cls, prompt in zip(class_names, text_prompts):
            print(f"  {cls}: {prompt}")
        print(f"{'='*80}\n")

    return {
        'class_names': class_names,
        'num_classes': num_classes,
        'dataset_name': dataset_display_name,
        'text_prompt_template': template,
        'text_prompts': text_prompts
    }


def load_class_names(dataset_name, class_names_dir=None):
    """
    Load class names from JSON configuration file (compatibility wrapper)

    Args:
        dataset_name: Name of the dataset (e.g., 'mimic_cxr', 'chestxray14')
        class_names_dir: Directory containing class name JSON files
                        (default: project_root/class_names/)

    Returns:
        List of class names

    Raises:
        FileNotFoundError: If class names file doesn't exist
        ValueError: If JSON file is invalid
    """
    config = load_class_config(dataset_name, class_names_dir, verbose=True)
    return config['class_names']


def list_available_datasets(class_names_dir=None):
    """
    List all available datasets with class name configurations

    Args:
        class_names_dir: Directory containing class name JSON files

    Returns:
        List of dataset names (without .json extension)
    """
    if class_names_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        class_names_dir = os.path.join(project_root, 'class_names')

    if not os.path.exists(class_names_dir):
        return []

    datasets = []
    for filename in os.listdir(class_names_dir):
        if filename.endswith('.json'):
            datasets.append(filename[:-5])  # Remove .json extension

    return sorted(datasets)
