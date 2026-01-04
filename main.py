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
    python main.py --epochs 50 --batch_size 32 --lr 5e-6
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
from utils import set_seed, print_device_info, load_class_names


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
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate for all parameters (default: 1e-5)')
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

    # ChestXray14 test set arguments
    parser.add_argument('--test_chestxray14', action='store_true',
                       help='Use ChestXray14 dataset for testing')
    parser.add_argument('--chestxray14_path', type=str, default=None,
                       help='Path to ChestXray14 dataset (default: D:\\Data\\ChestXray14\\CXR8)')

    # Testing control
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip testing after training (only train and save model)')

    # Zero-shot arguments
    parser.add_argument('--zeroshot_threshold', type=float, default=0.0,
                       help='Cosine similarity threshold for zero-shot multi-label classification (range: [-1, 1], higher=more conservative, default: 0.0)')

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
    if args.lr:
        config.training.learning_rate_image = args.lr
        config.training.learning_rate_text = args.lr
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


def run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader, experiment_name=None, threshold=0.0, class_names=None):
    """Run zero-shot CLIP inference"""
    print("\n" + "=" * 80)
    print("Starting Zero-shot CLIP")
    print("=" * 80)
    print(f"Using cosine similarity threshold: {threshold} (range: [-1, 1], higher=more conservative)")

    # Create zero-shot inference
    zeroshot = ZeroShotCLIPInference(clip_model, tokenizer, config)

    # Predict (pass class_names to use correct text prompts)
    all_predictions, all_labels, all_scores = zeroshot.predict(test_loader, threshold=threshold, class_names=class_names)

    # Generate save name
    save_name = experiment_name if experiment_name else "zeroshot"

    # Evaluate
    evaluator = ModelEvaluator(config, config.paths.output_dir)
    results = evaluator.evaluate(
        all_labels, all_predictions,
        "Zero-shot CLIP (MIMIC-CXR)",
        save_name,
        y_scores=all_scores,
        class_names=class_names
    )

    return results


def run_finetune(config, clip_model, tokenizer, preprocess, train_loader, val_loader, test_loader, train_labels=None, experiment_name=None, skip_test=False, class_names=None):
    """Run full fine-tuning with CLIP loss (supports multi-label)"""
    print("\n" + "=" * 80)
    print("Starting Fine-Tuning with CLIP Loss")
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
        model_name = 'finetune_best_model.pth'
        save_name = 'finetune'

    # Save best model
    trainer.save_best_model(model_name)

    # Skip testing if requested
    if skip_test:
        print("\n" + "=" * 80)
        print("Training completed! Skipping test evaluation as requested.")
        print(f"Model saved to: {config.paths.output_dir}/{model_name}")
        print("=" * 80)
        return {'best_val_recall_at_5': float(trainer.best_val_f1), 'skipped_test': True}  # best_val_f1 stores R@5 now

    # Predict on test set
    test_preds, test_labels = trainer.predict(test_loader)

    # For contrastive learning, we don't have classification predictions
    if test_preds is None or test_labels is None:
        print("\n" + "=" * 80)
        print("Contrastive learning completed!")
        print(f"Best validation Recall@5: {trainer.best_val_f1:.2f}%")
        print("=" * 80)
        return {'best_val_recall_at_5': float(trainer.best_val_f1), 'training_completed': True}

    # Evaluate (only if we have predictions)
    evaluator = ModelEvaluator(config, config.paths.output_dir)
    results = evaluator.evaluate(
        test_labels, test_preds,
        "Fine-Tuning - MIMIC-CXR",
        save_name,
        class_names=class_names
    )
    results['best_val_recall_at_5'] = float(trainer.best_val_f1)  # best_val_f1 stores R@5 now

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

    # Print configuration (only print training config if needed)
    if args.method in ['all', 'finetune']:
        config.print_config()
    else:
        # For zero-shot only, print minimal config
        print("=" * 80)
        print("Running Zero-shot Evaluation")
        print("=" * 80)
        print(f"Model: {config.model.model_name}")
        print(f"Output directory: {config.paths.output_dir}")
        print("=" * 80)

    # Load model
    model_loader = BiomedCLIPLoader(config)
    clip_model, tokenizer, preprocess = model_loader.load_model()

    # Wrap tokenizer
    tokenizer_wrapped = model_loader.create_tokenizer_wrapper(tokenizer)

    # Check if we need MIMIC data
    # Skip MIMIC loading if: zeroshot + ChestXray14 only
    need_mimic = not (args.method == 'zeroshot' and args.test_chestxray14)

    train_loader = None
    val_loader = None
    data_loader = None
    data_splits = None
    text_prompts = config.classes.get_text_prompts()

    if need_mimic:
        # Load and split MIMIC-CXR data
        print("\n" + "=" * 80)
        print("Loading MIMIC-CXR data...")
        print("=" * 80)

        data_loader = MIMICCXRDataLoader(config)
        image_paths, labels, reports, split_info = data_loader.load_data(
            use_provided_split=config.data.use_provided_split,
            label_policy=config.data.label_policy
        )
        data_splits = data_loader.split_dataset(image_paths, labels, reports, split_info)

    # Create data loaders
    if args.method in ['all', 'finetune']:
        # Create train/val loaders for fine-tuning (use reports)
        print("\nCreating fine-tuning loaders with radiology reports...")
        train_dataset_loader = data_loader.create_dataloaders(
            {'train': data_splits['train']},
            preprocess, tokenizer_wrapped, text_prompts, use_reports=True
        )
        val_dataset_loader = data_loader.create_dataloaders(
            {'val': data_splits['val']},
            preprocess, tokenizer_wrapped, text_prompts, use_reports=True
        )
        train_loader = train_dataset_loader['train']
        val_loader = val_dataset_loader['val']

    # Create test loader (MIMIC or ChestXray14)
    if args.test_chestxray14:
        print("\n" + "=" * 80)
        print("Using ChestXray14 TEST SET")
        print("=" * 80)

        # Load ChestXray14 test data
        external_data_path = args.chestxray14_path if args.chestxray14_path else r"D:\Data\ChestXray14\CXR8"
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
        # Use MIMIC-CXR test set (use text prompts for zero-shot evaluation)
        print("\nCreating test loader with text prompts (for zero-shot)...")
        test_dataset_loader = data_loader.create_dataloaders(
            {'test': data_splits['test']},
            preprocess, tokenizer_wrapped, text_prompts, use_reports=False
        )
        test_loader = test_dataset_loader['test']
        test_dataset_name = "MIMIC-CXR"

    # Automatically load class names based on test dataset
    if args.test_chestxray14:
        # Load ChestXray14 class names from config file
        eval_class_names = load_class_names('chestxray14')
    else:
        # Load MIMIC-CXR class names from config file
        eval_class_names = load_class_names('mimic_cxr')

    # Store all results
    all_results = []

    # Run methods
    if args.method in ['all', 'zeroshot']:
        experiment_suffix = f"_on_{test_dataset_name}" if args.test_chestxray14 else ""
        exp_name = (args.experiment_name + experiment_suffix) if args.experiment_name else f"zeroshot{experiment_suffix}"

        result1 = run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader,
                              experiment_name=exp_name, threshold=args.zeroshot_threshold,
                              class_names=eval_class_names)
        all_results.append(result1)

    if args.method in ['all', 'finetune']:
        # Reload model for fine-tuning
        if args.method == 'all':
            clip_model, tokenizer, preprocess = model_loader.load_model()

        # Get training labels for weighted/focal loss
        _, y_train = data_splits['train']

        experiment_suffix = f"_on_{test_dataset_name}" if args.test_chestxray14 else ""
        exp_name = (args.experiment_name + experiment_suffix) if args.experiment_name else f"finetune{experiment_suffix}"

        result2 = run_finetune(config, clip_model, tokenizer, preprocess,
                              train_loader, val_loader, test_loader,
                              train_labels=y_train, experiment_name=exp_name,
                              skip_test=args.skip_test, class_names=eval_class_names)
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
