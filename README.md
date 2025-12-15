# ISIC 2019 Skin Lesion Classification with BiomedCLIP

A modular PyTorch implementation for ISIC 2019 skin lesion classification using BiomedCLIP, supporting both zero-shot inference and full fine-tuning with CLIP loss.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, models, training, and evaluation
- **Two Training Methods**:
  - Method 1: Zero-shot CLIP inference
  - Method 2: Full fine-tuning with CLIP contrastive loss
- **Three Loss Functions** for handling class imbalance:
  - Standard CLIP Loss (baseline)
  - Weighted CLIP Loss (class re-weighting)
  - Focal CLIP Loss (hard example mining)
- **Flexible Configuration**: Easy-to-modify configuration system with command-line argument support
- **Comprehensive Evaluation**: Detailed metrics including accuracy, balanced accuracy, F1-scores, and per-class analysis
- **Visualization**: Automatic generation of confusion matrices, training curves, and class distribution plots

## Project Structure

```
isic_clip_project/
│
├── config/                      # Configuration module
│   ├── __init__.py
│   └── config.py               # Configuration classes (paths, model, data, training)
│
├── datasets/                    # Dataset module
│   ├── __init__.py
│   └── isic_dataset.py         # ISIC 2019 dataset class and data loader
│
├── models/                      # Model module
│   ├── __init__.py
│   └── clip_model.py           # CLIP model, loss function, and model loader
│
├── trainers/                    # Training module
│   ├── __init__.py
│   └── clip_trainer.py         # Zero-shot inference and fine-tuning trainer
│
├── evaluators/                  # Evaluation module
│   ├── __init__.py
│   └── evaluator.py            # Model evaluation and result comparison
│
├── utils/                       # Utility module
│   ├── __init__.py
│   └── helpers.py              # Helper functions (early stopping, seed setting, etc.)
│
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- BiomedCLIP checkpoint files

### Setup

1. Clone or download this project:
```bash
cd C:\Users\admin\Desktop\baseline\isic2019\isic_clip_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download ISIC 2019 dataset:
   - Download from [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/)
   - Place images in `D:\Data\isic2019\ISIC_2019_Training_Input\`
   - Place ground truth CSV in `D:\Data\isic2019\ISIC_2019_Training_GroundTruth.csv`

4. Ensure BiomedCLIP checkpoints are available at:
   ```
   C:\Users\admin\Desktop\baseline\bimedclip-zs\checkpoints\
   ├── open_clip_config.json
   ├── open_clip_pytorch_model.bin
   ├── tokenizer.json
   └── tokenizer_config.json
   ```

## Usage

### Basic Usage

Run with default configuration (both zero-shot and fine-tuning):
```bash
python main.py
```

### Run Specific Methods

Run only zero-shot inference:
```bash
python main.py --method zeroshot
```

Run only fine-tuning:
```bash
python main.py --method finetune
```

### Custom Paths

Specify custom data path:
```bash
python main.py --data_path D:\Data\my_isic2019
```

Specify custom checkpoint directory:
```bash
python main.py --checkpoint_dir C:\Models\biomedclip
```

Specify custom output directory:
```bash
python main.py --output_dir C:\Results\my_experiment
```

### Training Parameters

Modify batch size and epochs:
```bash
python main.py --batch_size 32 --epochs 50
```

Adjust learning rates:
```bash
python main.py --lr_image 5e-6 --lr_text 1e-4
```

Set early stopping patience:
```bash
python main.py --patience 15
```

### Data Split Configuration

Modify test/validation split:
```bash
python main.py --test_size 0.15 --val_size 0.15
```

### Loss Function Selection

The project supports three loss functions for handling class imbalance:

Use **Weighted CLIP Loss** (recommended for imbalanced datasets):
```bash
python main.py --method finetune --loss_type weighted
```

Use **Focal CLIP Loss** (focuses on hard examples):
```bash
python main.py --method finetune --loss_type focal --focal_gamma 2.0
```

Use **Focal Loss with class weights** (best for severe imbalance):
```bash
python main.py --method finetune --loss_type focal --focal_gamma 2.5 --focal_alpha
```

Customize class weight computation:
```bash
python main.py --loss_type weighted --class_weight_method effective
```

See `LOSS_FUNCTIONS_GUIDE.md` for detailed explanations and recommendations.

### Complete Example

```bash
python main.py \
    --method finetune \
    --data_path D:\Data\isic2019 \
    --output_dir C:\Results\experiment_001 \
    --batch_size 32 \
    --epochs 100 \
    --lr_image 1e-5 \
    --lr_text 1e-4 \
    --patience 10 \
    --seed 42
```

## Command-Line Arguments

### Method Selection
- `--method`: Choose method to run (`all`, `zeroshot`, `finetune`)

### Path Arguments
- `--data_path`: Path to ISIC 2019 data directory
- `--checkpoint_dir`: Path to BiomedCLIP checkpoint directory
- `--output_dir`: Path to output directory
- `--experiment_name`: Experiment name for saved files (customizes filenames)

### Data Arguments
- `--batch_size`: Batch size for training (default: 64)
- `--num_workers`: Number of data loading workers (default: 4)
- `--test_size`: Test set size ratio (default: 0.2)
- `--val_size`: Validation set size ratio (default: 0.1)

### Training Arguments
- `--epochs`: Number of training epochs (default: 100)
- `--lr_image`: Learning rate for image encoder (default: 1e-5)
- `--lr_text`: Learning rate for text embeddings (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--patience`: Early stopping patience (default: 10)

### Other Arguments
- `--seed`: Random seed for reproducibility (default: 42)
- `--gpu`: GPU ID to use

## Configuration

You can also modify the default configuration directly in `config/config.py`:

```python
from config import Config

# Create custom configuration
config = Config()

# Modify paths
config.paths.base_data_path = r"D:\Data\my_isic2019"
config.paths.output_dir = r"C:\Results\my_experiment"

# Modify training parameters
config.training.epochs = 50
config.training.learning_rate_image = 5e-6
config.data.batch_size = 32
```

## Output Files

The program generates the following outputs in the specified output directory:

### Visualizations
- `class_distribution.png`: Dataset class distribution bar chart
- `training_curves.png`: Training and validation loss/accuracy curves
- `method1_zeroshot_confusion_matrix.png`: Zero-shot confusion matrix
- `method2_full_finetune_confusion_matrix.png`: Fine-tuning confusion matrix
- `method1_zeroshot_per_class_recall.png`: Zero-shot per-class recall
- `method2_full_finetune_per_class_recall.png`: Fine-tuning per-class recall
- `methods_comparison.png`: Comparison of different methods

### Results
- `method1_zeroshot_results.json`: Zero-shot evaluation results
- `method2_full_finetune_results.json`: Fine-tuning evaluation results
- `methods_comparison.csv`: Comparison table of all methods

### Models
- `method2_best_model.pth`: Best fine-tuned model weights

## Dataset Information

### ISIC 2019 Classes (8 classes, UNK excluded)

| Class Code | Description |
|------------|-------------|
| MEL | Melanoma |
| NV | Melanocytic nevus |
| BCC | Basal cell carcinoma |
| AK | Actinic keratosis |
| BKL | Benign keratosis |
| DF | Dermatofibroma |
| VASC | Vascular lesion |
| SCC | Squamous cell carcinoma |

### Data Split

By default:
- Training: 70% of data
- Validation: 10% of data
- Test: 20% of data

All splits use stratified sampling to maintain class balance.

## Evaluation Metrics

The following metrics are computed:

- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Average recall across all classes (handles class imbalance)
- **F1-score (Macro)**: Unweighted average of per-class F1 scores
- **F1-score (Weighted)**: Weighted average by class support
- **Per-class F1**: Individual F1 score for each class
- **Per-class Recall**: Individual recall for each class
- **Confusion Matrix**: Full confusion matrix

## Model Details

### Method 1: Zero-shot CLIP
- Uses pre-trained BiomedCLIP without any training
- Compares image features with text prompt embeddings
- Text prompts: "this is a photo of {class_description}"

### Method 2: Full Fine-tuning with CLIP Loss
- Fine-tunes both image encoder and learnable text embeddings
- Uses contrastive CLIP loss (image-to-text and text-to-image)
- Separate learning rates for image encoder and text embeddings
- Early stopping based on validation F1-score (macro)
- Cosine annealing learning rate scheduler

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python main.py --batch_size 16
```

### Data Not Found
Check paths in configuration:
```bash
python main.py --data_path YOUR_DATA_PATH --checkpoint_dir YOUR_CHECKPOINT_PATH
```

### Slow Training
Reduce number of workers or use GPU:
```bash
python main.py --num_workers 2 --gpu 0
```

## Citation

If you use this code, please cite:

```bibtex
@misc{isic2019_biomedclip,
  title={ISIC 2019 Skin Lesion Classification with BiomedCLIP},
  author={Your Name},
  year={2024}
}
```

## License

This project is for research purposes only.

## Acknowledgments

- ISIC 2019 Challenge organizers for the dataset
- BiomedCLIP authors for the pre-trained model
- OpenAI for the original CLIP architecture
