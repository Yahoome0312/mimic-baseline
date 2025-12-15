"""
Configuration Example

This file demonstrates how to create and customize configurations programmatically.
"""

from config import Config

# Example 1: Use default configuration
def example_default_config():
    config = Config()
    config.print_config()
    return config


# Example 2: Modify paths
def example_custom_paths():
    config = Config()

    # Update paths
    config.update_paths(
        base_data_path=r"D:\Data\my_isic2019",
        output_dir=r"C:\Results\my_experiment",
        local_checkpoints_dir=r"C:\Models\biomedclip"
    )

    config.print_config()
    return config


# Example 3: Modify training parameters
def example_custom_training():
    config = Config()

    # Modify training configuration
    config.training.epochs = 50
    config.training.learning_rate_image = 5e-6
    config.training.learning_rate_text = 5e-5
    config.training.early_stopping_patience = 15

    # Modify data configuration
    config.data.batch_size = 32
    config.data.test_size = 0.15
    config.data.val_size = 0.15

    config.print_config()
    return config


# Example 4: Create config from dictionary
def example_from_dict():
    config_dict = {
        'paths': {
            'base_data_path': r'D:\Data\isic2019',
            'output_dir': r'C:\Results\experiment_001'
        },
        'data': {
            'batch_size': 32,
            'test_size': 0.2,
            'val_size': 0.1
        },
        'training': {
            'epochs': 100,
            'learning_rate_image': 1e-5,
            'learning_rate_text': 1e-4
        }
    }

    config = Config.from_dict(config_dict)
    config.print_config()
    return config


# Example 5: Get text prompts
def example_text_prompts():
    config = Config()

    # Default prompt template
    prompts = config.classes.get_text_prompts()
    print("Default prompts:")
    for cls, prompt in zip(config.classes.class_names, prompts):
        print(f"  {cls}: {prompt}")

    print("\n")

    # Custom prompt template
    custom_prompts = config.classes.get_text_prompts(
        template="a medical image showing {description}"
    )
    print("Custom prompts:")
    for cls, prompt in zip(config.classes.class_names, custom_prompts):
        print(f"  {cls}: {prompt}")

    return config


if __name__ == "__main__":
    print("=" * 80)
    print("Configuration Examples")
    print("=" * 80)

    print("\n### Example 1: Default Configuration ###\n")
    example_default_config()

    print("\n### Example 2: Custom Paths ###\n")
    example_custom_paths()

    print("\n### Example 3: Custom Training Parameters ###\n")
    example_custom_training()

    print("\n### Example 4: From Dictionary ###\n")
    example_from_dict()

    print("\n### Example 5: Text Prompts ###\n")
    example_text_prompts()
