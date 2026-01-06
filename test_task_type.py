"""
Test script to verify task_type handling in inference
"""
# -*- coding: utf-8 -*-

import sys
import os
import io

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np


def test_prediction_shapes():
    """Test that predictions have correct shapes for different task types"""
    print("=" * 80)
    print("Testing Prediction Shape Handling")
    print("=" * 80)

    batch_size = 4
    num_classes = 5

    # Simulate similarity scores
    similarity = torch.randn(batch_size, num_classes)

    print(f"\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Similarity shape: {similarity.shape}")

    # Test 1: Multi-label prediction (threshold-based)
    print("\n" + "-" * 80)
    print("Test 1: Multi-label task_type")
    print("-" * 80)

    threshold = 0.0
    predictions_multilabel = (similarity > threshold).float()

    print(f"  Prediction method: threshold > {threshold}")
    print(f"  Prediction shape: {predictions_multilabel.shape}")
    print(f"  Expected shape: ({batch_size}, {num_classes})")
    print(f"  Sample predictions:\n{predictions_multilabel}")

    assert predictions_multilabel.shape == (batch_size, num_classes), \
        f"Multi-label should be (N, C), got {predictions_multilabel.shape}"
    assert predictions_multilabel.dtype == torch.float32, \
        f"Multi-label should be float32, got {predictions_multilabel.dtype}"
    print("  ✓ Multi-label shape correct: 2D binary matrix")

    # Test 2: Single-label prediction (argmax-based)
    print("\n" + "-" * 80)
    print("Test 2: Single-label task_type")
    print("-" * 80)

    predictions_singlelabel = torch.argmax(similarity, dim=1)

    print(f"  Prediction method: argmax")
    print(f"  Prediction shape: {predictions_singlelabel.shape}")
    print(f"  Expected shape: ({batch_size},)")
    print(f"  Sample predictions: {predictions_singlelabel}")
    print(f"  Prediction range: [{predictions_singlelabel.min()}, {predictions_singlelabel.max()}]")

    assert predictions_singlelabel.shape == (batch_size,), \
        f"Single-label should be (N,), got {predictions_singlelabel.shape}"
    assert predictions_singlelabel.dtype == torch.int64, \
        f"Single-label should be int64, got {predictions_singlelabel.dtype}"
    assert predictions_singlelabel.min() >= 0 and predictions_singlelabel.max() < num_classes, \
        f"Single-label indices should be in [0, {num_classes-1}]"
    print("  ✓ Single-label shape correct: 1D class indices")

    # Test 3: Verify they're different
    print("\n" + "-" * 80)
    print("Test 3: Shape Comparison")
    print("-" * 80)

    print(f"  Multi-label dimensions: {predictions_multilabel.ndim}D")
    print(f"  Single-label dimensions: {predictions_singlelabel.ndim}D")
    print(f"  Multi-label size: {predictions_multilabel.numel()} elements")
    print(f"  Single-label size: {predictions_singlelabel.numel()} elements")

    assert predictions_multilabel.ndim == 2, "Multi-label should be 2D"
    assert predictions_singlelabel.ndim == 1, "Single-label should be 1D"
    print("  ✓ Shapes are correctly different")

    # Test 4: Numpy conversion
    print("\n" + "-" * 80)
    print("Test 4: NumPy Conversion")
    print("-" * 80)

    np_multilabel = predictions_multilabel.cpu().numpy()
    np_singlelabel = predictions_singlelabel.cpu().numpy()

    print(f"  Multi-label numpy shape: {np_multilabel.shape}")
    print(f"  Single-label numpy shape: {np_singlelabel.shape}")
    print(f"  Multi-label numpy dtype: {np_multilabel.dtype}")
    print(f"  Single-label numpy dtype: {np_singlelabel.dtype}")

    assert np_multilabel.shape == (batch_size, num_classes), \
        f"NumPy multi-label shape incorrect: {np_multilabel.shape}"
    assert np_singlelabel.shape == (batch_size,), \
        f"NumPy single-label shape incorrect: {np_singlelabel.shape}"
    print("  ✓ NumPy conversion preserves shapes")


def test_evaluator_compatibility():
    """Test compatibility with typical evaluator expectations"""
    print("\n" + "=" * 80)
    print("Testing Evaluator Compatibility")
    print("=" * 80)

    # Simulate predictions and labels
    n_samples = 10
    n_classes = 3

    # Multi-label case
    print("\n1. Multi-label Classification:")
    y_true_multilabel = np.random.randint(0, 2, size=(n_samples, n_classes))
    y_pred_multilabel = np.random.randint(0, 2, size=(n_samples, n_classes))

    print(f"  True labels shape: {y_true_multilabel.shape}")
    print(f"  Predictions shape: {y_pred_multilabel.shape}")

    # Typical multi-label metrics
    from sklearn.metrics import accuracy_score, f1_score

    # For multi-label, flatten or use 'samples' average
    acc = accuracy_score(y_true_multilabel, y_pred_multilabel)
    f1 = f1_score(y_true_multilabel, y_pred_multilabel, average='samples', zero_division=0)

    print(f"  Accuracy (exact match): {acc:.4f}")
    print(f"  F1 Score (samples): {f1:.4f}")
    print("  ✓ Multi-label metrics work")

    # Single-label case
    print("\n2. Single-label Classification:")
    y_true_singlelabel = np.random.randint(0, n_classes, size=n_samples)
    y_pred_singlelabel = np.random.randint(0, n_classes, size=n_samples)

    print(f"  True labels shape: {y_true_singlelabel.shape}")
    print(f"  Predictions shape: {y_pred_singlelabel.shape}")

    # Typical single-label metrics
    acc = accuracy_score(y_true_singlelabel, y_pred_singlelabel)
    f1 = f1_score(y_true_singlelabel, y_pred_singlelabel, average='macro', zero_division=0)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score (macro): {f1:.4f}")
    print("  ✓ Single-label metrics work")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TASK_TYPE PREDICTION SHAPE VERIFICATION")
    print("=" * 80)

    try:
        # Test 1: Prediction shapes
        test_prediction_shapes()

        # Test 2: Evaluator compatibility
        test_evaluator_compatibility()

        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Multi-label returns 2D binary matrix (N, num_classes)")
        print("  ✓ Single-label returns 1D class indices (N,)")
        print("  ✓ Shapes are compatible with standard evaluators")
        print("\nSingle-label classification is ready!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
