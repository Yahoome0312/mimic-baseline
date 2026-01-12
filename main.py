
import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import BiomedCLIPLoader, CLIPFineTune
from datasets import MIMICCXRDataLoader, ChestXray14DataLoader, ChestXDet10DataLoader, CheXpertDataLoader
from trainers import ZeroShotCLIPInference, CLIPTrainer
from evaluators import ModelEvaluator
from utils import set_seed, print_device_info, load_class_names, load_class_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MIMIC CLIP Training')

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

    # ChestXray14 test set arguments
    parser.add_argument('--test_chestxray14', action='store_true',
                       help='Use ChestXray14 dataset for testing')
    parser.add_argument('--chestxray14_path', type=str, default=None,
                       help='Path to ChestXray14 dataset (default: D:\\Data\\ChestXray14\\CXR8)')

    # ChestXDet10 test set arguments
    parser.add_argument('--test_chestxdet10', action='store_true',
                       help='Use ChestXDet10 dataset for testing')
    parser.add_argument('--chestxdet10_path', type=str, default=None,
                       help='Path to ChestXDet10 dataset (default: D:\\Data\\ChestXDet10)')

    # CheXpert test set arguments
    parser.add_argument('--test_chexpert', action='store_true',
                       help='Use CheXpert dataset for testing')
    parser.add_argument('--chexpert_path', type=str, default=None,
                       help='Path to CheXpert dataset (default: D:\\Data\\CheXpert\\CheXpert-v1.0-small)')

    # Testing control
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip testing after training (only train and save model)')
    parser.add_argument('--test_model_path', type=str, default=None,
                       help='Path to pretrained model (.pth file). If provided, skip training and test only.')

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

    return config


def run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader, experiment_name=None, threshold=0.0, class_names=None, text_prompts=None):
    """Run zero-shot CLIP inference"""
    print("\n" + "=" * 80)
    print("Starting Zero-shot CLIP")
    print("=" * 80)
    print(f"Using cosine similarity threshold: {threshold} (range: [-1, 1], higher=more conservative)")

    # Create zero-shot inference
    zeroshot = ZeroShotCLIPInference(clip_model, tokenizer, config)

    # Predict (pass class_names to use correct text prompts)
    all_predictions, all_labels, all_scores = zeroshot.predict(test_loader, threshold=threshold, class_names=class_names, text_prompts=text_prompts)

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


def run_finetune(config, clip_model, tokenizer, preprocess, train_loader, val_loader, test_loader, experiment_name=None, skip_test=False, class_names=None, text_prompts=None):
    """Run full fine-tuning with CLIP loss (multi-label)"""
    print("\n" + "=" * 80)
    print("Starting Fine-Tuning with CLIP Loss")
    print("=" * 80)

    # Create fine-tuned model (no class number limitation)
    model = CLIPFineTune(clip_model)

    # Create trainer
    trainer = CLIPTrainer(
        model,
        config,
        config.paths.output_dir,
        experiment_name=experiment_name,
        tokenizer=tokenizer
    )

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
        return {'best_val_loss': float(trainer.best_val_loss), 'skipped_test': True}

    # Predict on test set using zero-shot style classification
    test_preds, test_labels, test_scores = trainer.predict(test_loader, class_names=class_names, threshold=0.0, text_prompts=text_prompts)

    # Evaluate
    evaluator = ModelEvaluator(config, config.paths.output_dir)
    results = evaluator.evaluate(
        test_labels, test_preds,
        "Fine-Tuning - MIMIC-CXR",
        save_name,
        y_scores=test_scores,
        class_names=class_names
    )
    results['best_val_loss'] = float(trainer.best_val_loss)

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
    # Skip MIMIC loading if: zeroshot + (ChestXray14, ChestXDet10, or CheXpert) only
    need_mimic = not (args.method == 'zeroshot' and (args.test_chestxray14 or args.test_chestxdet10 or args.test_chexpert))

    train_loader = None
    val_loader = None
    data_loader = None
    data_splits = None

    # Load MIMIC-CXR class configuration (with text prompts)
    mimic_class_config = load_class_config('mimic_cxr', verbose=True)
    text_prompts = mimic_class_config['text_prompts']

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

    # Create test loader (MIMIC, ChestXray14, ChestXDet10, or CheXpert)
    if args.test_chestxray14:
        # Load ChestXray14 class configuration
        chestxray14_config = load_class_config('chestxray14', verbose=True)
        chestxray14_prompts = chestxray14_config['text_prompts']

        # Load ChestXray14 test data
        external_data_path = args.chestxray14_path if args.chestxray14_path else r"D:\Data\ChestXray14\CXR8"
        external_loader = ChestXray14DataLoader(config, data_path=external_data_path)
        external_image_paths, external_labels = external_loader.load_test_data()

        test_loader = external_loader.create_dataloader(
            external_image_paths, external_labels,
            preprocess, tokenizer_wrapped, chestxray14_prompts
        )
        test_dataset_name = "ChestXray14"
    elif args.test_chestxdet10:
        # Load ChestXDet10 class configuration
        chestxdet10_config = load_class_config('chestxdet10', verbose=True)
        chestxdet10_prompts = chestxdet10_config['text_prompts']

        # Load ChestXDet10 test data
        external_data_path = args.chestxdet10_path if args.chestxdet10_path else r"D:\Data\ChestXDet10"
        external_loader = ChestXDet10DataLoader(config, data_path=external_data_path)
        external_image_paths, external_labels = external_loader.load_test_data()

        test_loader = external_loader.create_dataloader(
            external_image_paths, external_labels,
            preprocess, tokenizer_wrapped, chestxdet10_prompts
        )
        test_dataset_name = "ChestXDet10"
    elif args.test_chexpert:
        # Load CheXpert class configuration
        chexpert_config = load_class_config('chexpert_5class', verbose=True)
        chexpert_prompts = chexpert_config['text_prompts']

        # Load CheXpert test data
        external_data_path = args.chexpert_path if args.chexpert_path else r"D:\Data\CheXpert\CheXpert-v1.0-small"
        external_loader = CheXpertDataLoader(config, data_path=external_data_path)
        external_image_paths, external_labels = external_loader.load_test_data()

        test_loader = external_loader.create_dataloader(
            external_image_paths, external_labels,
            preprocess, tokenizer_wrapped, chexpert_prompts
        )
        test_dataset_name = "CheXpert"
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
    elif args.test_chestxdet10:
        # Load ChestXDet10 class names from config file
        eval_class_names = load_class_names('chestxdet10')
    elif args.test_chexpert:
        # Load CheXpert 5-class names from config file
        eval_class_names = load_class_names('chexpert_5class')
    else:
        # Load MIMIC-CXR class names from config file
        eval_class_names = load_class_names('mimic_cxr')

    # Automatically select text prompts based on test dataset
    if args.test_chestxray14:
        eval_text_prompts = chestxray14_prompts
    elif args.test_chestxdet10:
        eval_text_prompts = chestxdet10_prompts
    elif args.test_chexpert:
        eval_text_prompts = chexpert_prompts
    else:
        eval_text_prompts = text_prompts

    # Store all results
    all_results = []

    # Handle model loading for testing (if test_model_path is provided)
    if args.test_model_path:
        print("\n" + "=" * 80)
        print("LOADING PRETRAINED MODEL FOR TESTING")
        print("=" * 80)

        # Validate model path
        if not os.path.exists(args.test_model_path):
            raise FileNotFoundError(f"Model file not found: {args.test_model_path}")

        print(f"Model path: {args.test_model_path}")

        # Create CLIPFineTune model (no class limitation)
        from models import CLIPFineTune
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CLIPFineTune(clip_model)

        # Load model weights
        print(f"Loading model on {device}...")
        state_dict = torch.load(args.test_model_path, map_location=device)

        # Remove legacy text_embeddings if exists (from old checkpoints)
        if 'text_embeddings' in state_dict:
            print("Removing legacy text_embeddings from checkpoint...")
            del state_dict['text_embeddings']

        # Load weights (strict=False allows missing text_embeddings)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        print(f"✓ Model can test on ANY dataset (flexible inference)")

        # Create trainer for predict()
        from trainers import CLIPTrainer
        trainer = CLIPTrainer(
            model, config, config.paths.output_dir,
            experiment_name=args.experiment_name,
            tokenizer=tokenizer
        )
        trainer.best_model_path = args.test_model_path

        # Run prediction
        print("\n" + "=" * 80)
        print("Running Inference on Test Set")
        print("=" * 80)
        test_preds, test_labels, test_scores = trainer.predict(
            test_loader,
            class_names=eval_class_names,
            threshold=0.0,
            text_prompts=eval_text_prompts
        )

        # Evaluate
        from evaluators import ModelEvaluator
        evaluator = ModelEvaluator(config, config.paths.output_dir)
        save_name = args.experiment_name if args.experiment_name else "loaded_model_test"
        results = evaluator.evaluate(
            test_labels, test_preds,
            f"Loaded Model Test - {test_dataset_name}",
            save_name,
            y_scores=test_scores,
            class_names=eval_class_names
        )

        print("\n" + "=" * 80)
        print("TESTING COMPLETED")
        print("=" * 80)
        return results

    # Run methods
    if args.method in ['all', 'zeroshot']:
        # Use experiment_name as-is, or default to "zeroshot"
        # Model name reflects the training dataset (MIMIC), not test dataset
        exp_name = args.experiment_name if args.experiment_name else "zeroshot"

        result1 = run_zeroshot(config, clip_model, tokenizer, preprocess, test_loader,
                              experiment_name=exp_name, threshold=args.zeroshot_threshold,
                              class_names=eval_class_names, text_prompts=eval_text_prompts)
        all_results.append(result1)

    if args.method in ['all', 'finetune']:
        # Reload model for fine-tuning
        if args.method == 'all':
            clip_model, tokenizer, preprocess = model_loader.load_model()

        # Use experiment_name as-is, or default to "finetune"
        # Model name reflects the training dataset (MIMIC), not test dataset
        exp_name = args.experiment_name if args.experiment_name else "finetune"

        # Use test dataset class names for inference (dimension must match labels)
        result2 = run_finetune(config, clip_model, tokenizer, preprocess,
                              train_loader, val_loader, test_loader,
                              experiment_name=exp_name,
                              skip_test=args.skip_test,
                              class_names=eval_class_names,      # Test dataset classes
                              text_prompts=eval_text_prompts)    # Test dataset prompts
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
