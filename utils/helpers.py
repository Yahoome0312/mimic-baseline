"""
Utility helper functions
"""

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
                print(f"✓ Performance improved! New best_score = {score:.4f}")
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"✗ No improvement ({self.counter}/{self.patience})")

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
