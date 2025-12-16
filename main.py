"""
Main entry point for ISIC 2019 CLIP training

Usage examples:
    # Run with default configuration
    python main.py

    # Run only zero-shot
    python main.py --method zeroshot

    # Run only fine-tuning
    python main.py --method finetune

    # Specify custom data path and output directory
    python main.py --data_path D:\Data\isic2019 --output_dir C:\Results\my_experiment

    # Specify custom checkpoint directory
    python main.py --checkpoint_dir C:\Models\biomedclip

    # Modify training parameters
    python main.py --epochs 50 --batch_size 32 --lr_image 5e-6
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import BiomedCLIPLoader, CLIPFineTune
from datasets import MIMICCXRDataLoader, ChestXray14DataLoader
from trainers import ZeroShotCLIPInference, CLIPTrainer
from evaluators import ModelEvaluator
from utils import set_seed, print_device_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ISIC 2019 CLIP Training')

    # Method selection
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'zeroshot', 'finetune'],
                       help='Method to run: all, zeroshot, or finetune (default: all)')

    # Path arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to ISIC 2019 data directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to BiomedCLIP checkpoint directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for saved files (e.g., "weighted_exp1")')

    # Data arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--test_size', type=float, default=None,
                       help='Test set size (0-1)')
    parser.add_argument('--val_size', type=float, default=None,
                       help='Validation set size (0-1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--lr_image', type=float, default=None,
                       help='Learning rate for image encoder')
    parser.add_argument('--lr_text', type=float, default=None,
                       help='Learning rate for text embeddings')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience')

    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default=None,
                       choices=['standard', 'weighted', 'focal'],
                       help='Loss function type: standard, weighted, or focal')
    parser.add_argument('--class_weight_method', type=str, default=None,
                       choices=['inverse', 'sqrt_inverse', 'effective'],
                       help='Method for computing class weights')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal loss gamma parameter (focus on hard examples)')
    parser.add_argument('--focal_alpha', action='store_true',
                       help='Use class weights with focal loss')

    # External test set arguments
    parser.add_argument('--external_test', action='store_true',
                       help='Use external test set (ChestXray14) for evaluation')
    parser.add_argument('--external_test_path', type=str, default=None,
                       help='Path to external test dataset (ChestXray14)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use')

    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration from command line arguments"""

    # Update paths
    if args.data_path:
        config.update_paths(base_data_path=args.data_path)
    if args.checkpoint_dir:
        config.update_paths(local_checkpoints_dir=args.checkpoint_dir)
    if args.output_dir:
        config.update_paths(output_dir=args.output_dir)

    # Update data config
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.num_workers:
        config.data.num_workers = args.num_workers
    if args.test_size:
        config.data.test_size = args.test_size
    if args.val_size:
        config.data.val_size = args.val_size

    # Update training config
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr_image:
        config.training.learning_rate_image = args.lr_image
    if args.lr_text:
        config.training.learning_rate_text = args.lr_text
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay
    if args.patience:
        config.training.early_stopping_patience = args.patience

    # Update loss function config
    if args.loss_type:
        config.training.loss_type = args.loss_type
    if args.class_weight_method:
        config.training.class_weight_method = args.class_weight_method
    if args.focal_gamma:
        config.training.focal_gamma = args.focal_gamma
    if args.focal_alpha:
        config.training.focal_alpha = True

    return config


def run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader, experiment_name=None):
    """Run zero-shot CLIP inference"""
    print("\n" + "=" * 80)
    print("Starting Method 1: Zero-shot CLIP")
    print("=" * 80)

    # Create zero-shot inference
    zeroshot = ZeroShotCLIPInference(clip_model, tokenizer, config)

    # Predict
    all_predictions, all_labels, all_scores = zeroshot.predict(test_loader)

    # Generate save name
    save_name = experiment_name if experiment_name else "method1_zeroshot"

    # Evaluate
    evaluator = ModelEvaluator(config, config.paths.output_dir)
    results = evaluator.evaluate(
        all_labels, all_predictions,
        "Method 1: Zero-shot CLIP (MIMIC-CXR)",
        save_name,
        y_scores=all_scores
    )

    return results


def run_finetune(config, clip_model, tokenizer, preprocess, train_loader, val_loader, test_loader, train_labels=None, experiment_name=None):
    """Run full fine-tuning with CLIP loss (supports multi-label)"""
    print("\n" + "=" * 80)
    print("Starting Method 2: Full Fine-Tuning with CLIP/BCE Loss")
    print("=" * 80)

    # Create fine-tuned model
    model = CLIPFineTune(clip_model, config.classes.num_classes)

    # Create trainer (pass train_labels for weighted/focal losses)
    trainer = CLIPTrainer(model, config, config.paths.output_dir,
                         train_labels=train_labels, experiment_name=experiment_name)

    # Train
    trainer.train(train_loader, val_loader)

    # Generate save names
    if experiment_name:
        model_name = f'{experiment_name}_best_model.pth'
        save_name = experiment_name
    else:
        model_name = 'method2_best_model.pth'
        save_name = 'method2_full_finetune'

    # Save best model
    trainer.save_best_model(model_name)

    # Predict on test set
    test_preds, test_labels = trainer.predict(test_loader)

    # Evaluate
    evaluator = ModelEvaluator(config, config.paths.output_dir)
    results = evaluator.evaluate(
        test_labels, test_preds,
        "Method 2: Full Fine-Tuning - MIMIC-CXR",
        save_name
    )
    results['best_val_f1_macro'] = float(trainer.best_val_f1)

    return results


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Print device info
    print_device_info()

    # Create configuration
    config = Config()

    # Update configuration from arguments
    config = update_config_from_args(config, args)

    # Print configuration
    config.print_config()

    # Load model
    model_loader = BiomedCLIPLoader(config)
    clip_model, tokenizer, preprocess = model_loader.load_model()

    # Wrap tokenizer
    tokenizer_wrapped = model_loader.create_tokenizer_wrapper(tokenizer)

    # Load and split MIMIC-CXR data
    data_loader = MIMICCXRDataLoader(config)
    image_paths, labels, split_info = data_loader.load_data(
        use_provided_split=config.data.use_provided_split,
        label_policy=config.data.label_policy
    )
    data_splits = data_loader.split_dataset(image_paths, labels, split_info)

    # Get text prompts
    text_prompts = config.classes.get_text_prompts()

    # Create data loaders
    print("\n" + "=" * 80)
    print("Creating data loaders...")
    print("=" * 80)

    train_loader = None
    val_loader = None

    if args.method in ['all', 'finetune']:
        # Create train/val loaders for fine-tuning
        train_dataset_loader = data_loader.create_dataloaders(
            {'train': data_splits['train']},
            preprocess, tokenizer_wrapped, text_prompts
        )
        val_dataset_loader = data_loader.create_dataloaders(
            {'val': data_splits['val']},
            preprocess, tokenizer_wrapped, text_prompts
        )
        train_loader = train_dataset_loader['train']
        val_loader = val_dataset_loader['val']

    # Create test loader (MIMIC or ChestXray14)
    if args.external_test:
        print("\n" + "=" * 80)
        print("Using EXTERNAL TEST SET: ChestXray14")
        print("=" * 80)

        # Load ChestXray14 test data
        external_data_path = args.external_test_path if args.external_test_path else r"D:\Data\ChestXray14\CXR8"
        external_loader = ChestXray14DataLoader(config, data_path=external_data_path)
        external_image_paths, external_labels = external_loader.load_test_data()

        # Use ChestXray14 class names for text prompts
        chestxray14_prompts = [f"chest x-ray showing {cls.lower().replace('_', ' ')}"
                               for cls in external_loader.CHESTXRAY14_CLASSES]

        test_loader = external_loader.create_dataloader(
            external_image_paths, external_labels,
            preprocess, tokenizer_wrapped, chestxray14_prompts
        )
        test_dataset_name = "ChestXray14"
    else:
        # Use MIMIC-CXR test set
        test_dataset_loader = data_loader.create_dataloaders(
            {'test': data_splits['test']},
            preprocess, tokenizer_wrapped, text_prompts
        )
        test_loader = test_dataset_loader['test']
        test_dataset_name = "MIMIC-CXR"

    # Store all results
    all_results = []

    # Run methods
    if args.method in ['all', 'zeroshot']:
        experiment_suffix = f"_on_{test_dataset_name}" if args.external_test else ""
        exp_name = (args.experiment_name + experiment_suffix) if args.experiment_name else f"method1_zeroshot{experiment_suffix}"

        result1 = run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader,
                              experiment_name=exp_name)
        all_results.append(result1)

    if args.method in ['all', 'finetune']:
        # Reload model for fine-tuning
        if args.method == 'all':
            clip_model, tokenizer, preprocess = model_loader.load_model()

        # Get training labels for weighted/focal loss
        _, y_train = data_splits['train']

        experiment_suffix = f"_on_{test_dataset_name}" if args.external_test else ""
        exp_name = (args.experiment_name + experiment_suffix) if args.experiment_name else f"method2_full_finetune{experiment_suffix}"

        result2 = run_finetune(config, clip_model, tokenizer, preprocess,
                              train_loader, val_loader, test_loader,
                              train_labels=y_train, experiment_name=exp_name)
        all_results.append(result2)

    # Compare results if multiple methods were run
    if len(all_results) > 1:
        evaluator = ModelEvaluator(config, config.paths.output_dir)
        evaluator.compare_methods(all_results)

    print("\n" + "=" * 80)
    print("All methods completed!")
    print("=" * 80)
    print(f"All results saved to: {config.paths.output_dir}")


if __name__ == "__main__":
    main()
