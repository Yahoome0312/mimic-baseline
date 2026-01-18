import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathConfig:
    """Path configuration"""
    # Model checkpoint path
    local_checkpoints_dir: str = r"C:\Users\admin\Desktop\mimic-baseline\checkpoints"

    # Data path - Changed to MIMIC-CXR dataset
    base_data_path: str = r"D:\Data\MIMIC"

    # Output path - Updated for MIMIC results
    output_dir: str = r"C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip"

    # Tokenization cache path (defaults to base_data_path/tokenized_cache)
    tokenized_cache_dir: str = None

    def __post_init__(self):
        """Create output directory and cache directory if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Set default cache directory if not provided
        if self.tokenized_cache_dir is None:
            # Place cache inside base_data_path
            self.tokenized_cache_dir = os.path.join(self.base_data_path, 'tokenized_cache')

        # Create cache directory
        os.makedirs(self.tokenized_cache_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration"""
    # Use local-dir schema to load tokenizer from local directory
    model_name: str = r"local-dir:C:\Users\admin\Desktop\mimic-baseline\checkpoints"
    context_length: int = 256
    temperature: float = 0.07  # CLIP loss temperature


@dataclass
class DataConfig:
    """Data configuration"""
    batch_size: int = 96  
    num_workers: int = 8  
    val_size: float = 0.1
    random_state: int = 42
    use_provided_split: bool = True  # Use official MIMIC-CXR split
    label_policy: str = 'ignore_uncertain'  # How to handle uncertain labels in MIMIC


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Learning rates
    learning_rate_image: float = 5e-5
    learning_rate_text: float = 5e-5  # Changed to same as image encoder

    # Training parameters
    epochs: int = 10
    weight_decay: float = 0.01
    early_stopping_patience: int = 10

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, or none

    # Warmup
    use_warmup: bool = True
    warmup_epochs: int = 5          # Number of epochs for warmup
    warmup_start_factor: float = 0.01  # Starting LR = 1% of base LR
    warmup_end_factor: float = 1.0     # Ending LR = 100% of base LR



@dataclass
class Config:
    """Main configuration class"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )

    def update_paths(self, **kwargs):
        """Update path configuration"""
        for key, value in kwargs.items():
            if hasattr(self.paths, key):
                setattr(self.paths, key, value)
        # Refresh cache dir when base_data_path changes unless explicitly set.
        if 'base_data_path' in kwargs and 'tokenized_cache_dir' not in kwargs:
            self.paths.tokenized_cache_dir = os.path.join(
                self.paths.base_data_path, 'tokenized_cache'
            )
        # Recreate output and cache directories
        os.makedirs(self.paths.output_dir, exist_ok=True)
        if self.paths.tokenized_cache_dir:
            os.makedirs(self.paths.tokenized_cache_dir, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'paths': self.paths.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__
        }

    def print_config(self):
        """Print configuration"""
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        print("\n[Paths]")
        for key, value in self.paths.__dict__.items():
            print(f"  {key}: {value}")

        print("\n[Model]")
        for key, value in self.model.__dict__.items():
            print(f"  {key}: {value}")

        print("\n[Data]")
        for key, value in self.data.__dict__.items():
            print(f"  {key}: {value}")

        print("\n[Training]")
        for key, value in self.training.__dict__.items():
            print(f"  {key}: {value}")

        print("=" * 80)


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.print_config()

    # Test updating paths
    print("\n\nUpdating paths...")
    config.update_paths(
        base_data_path=r"D:\Data\new_isic2019",
        output_dir=r"C:\Results\test"
    )
    config.print_config()
