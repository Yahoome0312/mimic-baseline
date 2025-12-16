"""
Configuration file for ISIC 2019 CLIP training project
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathConfig:
    """Path configuration"""
    # Model checkpoint path
    local_checkpoints_dir: str = r"C:\Users\admin\Desktop\baseline\bimedclip-zs\checkpoints"

    # Data path - Changed to MIMIC-CXR dataset
    base_data_path: str = r"D:\Data\MIMIC"

    # Output path - Updated for MIMIC results
    output_dir: str = r"C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip"

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "biomedclip_local"
    context_length: int = 256
    temperature: float = 0.07  # CLIP loss temperature


@dataclass
class DataConfig:
    """Data configuration"""
    batch_size: int = 32  # Reduced for MIMIC-CXR (larger images)
    num_workers: int = 4
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    use_provided_split: bool = True  # Use official MIMIC-CXR split
    label_policy: str = 'ignore_uncertain'  # How to handle uncertain labels in MIMIC


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Learning rates
    learning_rate_image: float = 1e-5
    learning_rate_text: float = 1e-4

    # Training parameters
    epochs: int = 100
    weight_decay: float = 0.01
    early_stopping_patience: int = 10

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, or none

    # Loss function configuration
    loss_type: str = "standard"  # standard, weighted, focal
    use_class_weights: bool = False
    class_weight_method: str = "inverse"  # inverse, sqrt_inverse, effective
    focal_gamma: float = 2.0  # focal loss focusing parameter (only for focal loss)
    focal_alpha: bool = True  # whether to use class weights with focal loss


@dataclass
class ClassConfig:
    """Class configuration for MIMIC-CXR (CheXpert labels)"""
    # Class descriptions (14 CheXpert classes for chest X-ray)
    class_descriptions: Dict[str, str] = field(default_factory=lambda: {
        'Atelectasis': 'atelectasis',
        'Cardiomegaly': 'cardiomegaly',
        'Consolidation': 'consolidation',
        'Edema': 'edema',
        'Enlarged Cardiomediastinum': 'enlarged cardiomediastinum',
        'Fracture': 'fracture',
        'Lung Lesion': 'lung lesion',
        'Lung Opacity': 'lung opacity',
        'No Finding': 'no finding',
        'Pleural Effusion': 'pleural effusion',
        'Pleural Other': 'pleural abnormality',
        'Pneumonia': 'pneumonia',
        'Pneumothorax': 'pneumothorax',
        'Support Devices': 'support devices',
    })

    # Task type: 'multi-label' for MIMIC-CXR, 'single-label' for ISIC
    task_type: str = 'multi-label'

    @property
    def class_names(self) -> List[str]:
        """Get list of class names"""
        return list(self.class_descriptions.keys())

    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return len(self.class_descriptions)

    def get_text_prompts(self, template: str = "chest x-ray showing {description}") -> List[str]:
        """
        Generate text prompts for each class

        Args:
            template: Prompt template, use {description} as placeholder
                     Default changed to chest X-ray specific template

        Returns:
            List of text prompts
        """
        return [template.format(description=desc) for desc in self.class_descriptions.values()]


@dataclass
class Config:
    """Main configuration class"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            classes=ClassConfig(**config_dict.get('classes', {}))
        )

    def update_paths(self, **kwargs):
        """Update path configuration"""
        for key, value in kwargs.items():
            if hasattr(self.paths, key):
                setattr(self.paths, key, value)
        # Recreate output directory
        os.makedirs(self.paths.output_dir, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'paths': self.paths.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'classes': {
                'class_descriptions': self.classes.class_descriptions
            }
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

        print("\n[Classes]")
        print(f"  num_classes: {self.classes.num_classes}")
        print(f"  class_names: {self.classes.class_names}")
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
