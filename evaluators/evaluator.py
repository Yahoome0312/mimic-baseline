"""
Evaluation Module
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    hamming_loss,
    jaccard_score
)


class ModelEvaluator:
    """Model evaluation class"""

    def __init__(self, config, output_dir):
        """
        Args:
            config: Configuration object
            output_dir: Output directory for saving results
        """
        self.config = config
        self.output_dir = output_dir
        self.class_names = config.classes.class_names
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, method_name, save_path=None, y_scores=None):
        """
        Evaluate model performance (supports both single-label and multi-label)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            method_name: Method name for display
            save_path: Prefix for saved files
            y_scores: Prediction scores (for AUC calculation in multi-label)

        Returns:
            Dictionary containing evaluation results
        """
        print("\n" + "=" * 80)
        print(f"{method_name} - Evaluation Results")
        print("=" * 80)

        is_multilabel = self.config.classes.task_type == 'multi-label'

        if is_multilabel:
            return self._evaluate_multilabel(y_true, y_pred, method_name, save_path, y_scores)
        else:
            return self._evaluate_singlelabel(y_true, y_pred, method_name, save_path)

    def _evaluate_singlelabel(self, y_true, y_pred, method_name, save_path):
        """Evaluate single-label classification"""
        # Calculate various metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Calculate F1 score for each class
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1-score (Macro): {f1_macro:.4f}")
        print(f"  F1-score (Weighted): {f1_weighted:.4f}")

        print(f"\nF1-score per class:")
        for i, cls in enumerate(self.class_names):
            print(f"  {cls}: {f1_per_class[i]:.4f}")

        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4, zero_division=0)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Visualize confusion matrix
        if save_path:
            self._plot_confusion_matrix(cm, method_name, save_path)

        # Calculate and plot per-class recall
        per_class_recall, per_class_count = self._calculate_per_class_recall(y_true, y_pred)

        if save_path:
            self._plot_per_class_recall(per_class_recall, per_class_count, accuracy, balanced_acc,
                                       method_name, save_path)

        # Aggregate results
        results = {
            'method': method_name,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': {self.class_names[i]: float(f1_per_class[i]) for i in range(len(self.class_names))},
            'recall_per_class': per_class_recall,
            'samples_per_class': per_class_count,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        }

        if save_path:
            with open(os.path.join(self.output_dir, f'{save_path}_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def _evaluate_multilabel(self, y_true, y_pred, method_name, save_path, y_scores=None):
        """Evaluate multi-label classification"""
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Overall multi-label metrics
        subset_accuracy = accuracy_score(y_true, y_pred)  # Exact match ratio
        hamming = hamming_loss(y_true, y_pred)  # Hamming loss
        jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)  # Jaccard similarity

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Macro/micro averages
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # AUC scores (if scores provided)
        auc_per_class = {}
        auc_macro = None
        if y_scores is not None:
            y_scores = np.array(y_scores)
            try:
                for i, cls in enumerate(self.class_names):
                    if y_true[:, i].sum() > 0:  # Only if class has positive samples
                        auc_per_class[cls] = roc_auc_score(y_true[:, i], y_scores[:, i])
                    else:
                        auc_per_class[cls] = 0.0

                # Try to calculate macro AUC
                auc_macro = roc_auc_score(y_true, y_scores, average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate AUC scores: {e}")
                auc_per_class = {cls: 0.0 for cls in self.class_names}

        # Print results
        print(f"\nOverall Multi-Label Metrics:")
        print(f"  Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  Jaccard Score (Samples Avg): {jaccard:.4f}")
        print(f"  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (Macro): {recall_macro:.4f}")
        print(f"  F1-score (Macro): {f1_macro:.4f}")
        print(f"  Precision (Micro): {precision_micro:.4f}")
        print(f"  Recall (Micro): {recall_micro:.4f}")
        print(f"  F1-score (Micro): {f1_micro:.4f}")
        if auc_macro is not None:
            print(f"  AUC-ROC (Macro): {auc_macro:.4f}")

        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12} {'Support':<12}")
        print("=" * 100)
        for i, cls in enumerate(self.class_names):
            support = int(y_true[:, i].sum())
            auc_str = f"{auc_per_class.get(cls, 0.0):.4f}" if auc_per_class else "N/A"
            print(f"{cls:<30} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
                  f"{f1_per_class[i]:<12.4f} {auc_str:<12} {support:<12}")

        # Plot per-class metrics
        if save_path:
            self._plot_multilabel_metrics(
                precision_per_class, recall_per_class, f1_per_class,
                y_true, method_name, save_path
            )

        # Aggregate results
        results = {
            'method': method_name,
            'task_type': 'multi-label',
            'subset_accuracy': float(subset_accuracy),
            'hamming_loss': float(hamming),
            'jaccard_score': float(jaccard),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_per_class': {self.class_names[i]: float(precision_per_class[i]) for i in range(len(self.class_names))},
            'recall_per_class': {self.class_names[i]: float(recall_per_class[i]) for i in range(len(self.class_names))},
            'f1_per_class': {self.class_names[i]: float(f1_per_class[i]) for i in range(len(self.class_names))},
            'support_per_class': {self.class_names[i]: int(y_true[:, i].sum()) for i in range(len(self.class_names))},
        }

        if auc_macro is not None:
            results['auc_macro'] = float(auc_macro)
            results['auc_per_class'] = auc_per_class

        if save_path:
            with open(os.path.join(self.output_dir, f'{save_path}_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def _plot_multilabel_metrics(self, precision, recall, f1, y_true, method_name, save_path):
        """Plot multi-label per-class metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Support per class
        support = [int(y_true[:, i].sum()) for i in range(len(self.class_names))]

        # Precision
        axes[0].barh(self.class_names, precision, color='skyblue')
        axes[0].set_xlabel('Precision')
        axes[0].set_title('Precision per Class')
        axes[0].set_xlim([0, 1])
        for i, (p, s) in enumerate(zip(precision, support)):
            axes[0].text(p + 0.02, i, f'{p:.3f} (n={s})', va='center')

        # Recall
        axes[1].barh(self.class_names, recall, color='lightcoral')
        axes[1].set_xlabel('Recall')
        axes[1].set_title('Recall per Class')
        axes[1].set_xlim([0, 1])
        for i, (r, s) in enumerate(zip(recall, support)):
            axes[1].text(r + 0.02, i, f'{r:.3f} (n={s})', va='center')

        # F1-score
        axes[2].barh(self.class_names, f1, color='lightgreen')
        axes[2].set_xlabel('F1-Score')
        axes[2].set_title('F1-Score per Class')
        axes[2].set_xlim([0, 1])
        for i, (f, s) in enumerate(zip(f1, support)):
            axes[2].text(f + 0.02, i, f'{f:.3f} (n={s})', va='center')

        fig.suptitle(f'{method_name} - Per-Class Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{save_path}_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-label metrics plot saved to: {save_path}_metrics.png")

    def _plot_confusion_matrix(self, cm, method_name, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{method_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{save_path}_confusion_matrix.png'))
        plt.close()
        print(f"Confusion matrix saved to: {save_path}_confusion_matrix.png")

    def _calculate_per_class_recall(self, y_true, y_pred):
        """Calculate per-class recall"""
        per_class_recall = {}
        per_class_count = {}

        for i, cls in enumerate(self.class_names):
            # Find all samples of this class
            cls_indices = [j for j, label in enumerate(y_true) if label == i]
            per_class_count[cls] = len(cls_indices)

            if len(cls_indices) > 0:
                # Calculate recall for this class: TP / (TP + FN)
                cls_true = [y_true[j] for j in cls_indices]
                cls_pred = [y_pred[j] for j in cls_indices]
                per_class_recall[cls] = accuracy_score(cls_true, cls_pred)  # This is actually Recall!
            else:
                per_class_recall[cls] = 0.0

        return per_class_recall, per_class_count

    def _plot_per_class_recall(self, per_class_recall, per_class_count, accuracy, balanced_acc,
                               method_name, save_path):
        """Plot per-class recall"""
        # Sort by sample count (consistent with class distribution plot)
        class_data = [(cls, per_class_count[cls], per_class_recall[cls]) for cls in self.class_names]
        class_data_sorted = sorted(class_data, key=lambda x: x[1], reverse=True)
        sorted_names = [item[0] for item in class_data_sorted]
        sorted_counts = [item[1] for item in class_data_sorted]
        sorted_recalls = [item[2] for item in class_data_sorted]

        fig, ax = plt.subplots(figsize=(14, 7))

        # Use colors to represent recall level
        colors = plt.cm.RdYlGn(np.array(sorted_recalls))

        # Bar chart
        bars = ax.bar(range(len(sorted_names)), sorted_recalls,
                      color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)

        # Add recall labels on bars
        for i, (bar, recall, count) in enumerate(zip(bars, sorted_recalls, sorted_counts)):
            height = bar.get_height()
            # Recall label
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{recall:.2%}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            # Sample count label
            ax.text(bar.get_x() + bar.get_width()/2., -0.05,
                    f'n={count}',
                    ha='center', va='top', fontsize=9, style='italic')

        # Set x-axis labels
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=11)

        # Set labels and title
        ax.set_xlabel('Class (Sorted by Sample Count)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=13, fontweight='bold')
        ax.set_title(f'{method_name} - Per-Class Recall (TP/(TP+FN))', fontsize=16, fontweight='bold', pad=20)

        # Set y-axis range
        ax.set_ylim([0, 1.1])

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Set borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        # Add overall info box
        overall_text = f'Overall Accuracy: {accuracy:.2%}\nBalanced Accuracy (=Mean Recall): {balanced_acc:.2%}'
        ax.text(0.02, 0.97, overall_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2))

        plt.tight_layout()

        # Save figure
        recall_plot_path = os.path.join(self.output_dir, f'{save_path}_per_class_recall.png')
        plt.savefig(recall_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class recall plot saved to: {save_path}_per_class_recall.png")

    def compare_methods(self, all_results):
        """
        Compare results of multiple methods

        Args:
            all_results: List of result dictionaries from different methods
        """
        print("\n" + "=" * 80)
        print("Comparison of Methods")
        print("=" * 80)

        # Create comparison table
        comparison_df = pd.DataFrame({
            'Method': [r['method'] for r in all_results],
            'Accuracy': [r['accuracy'] for r in all_results],
            'Balanced Accuracy': [r['balanced_accuracy'] for r in all_results],
            'F1-Macro': [r['f1_macro'] for r in all_results],
            'F1-Weighted': [r['f1_weighted'] for r in all_results],
        })

        print("\n" + str(comparison_df))

        # Save comparison results
        comparison_df.to_csv(os.path.join(self.output_dir, 'methods_comparison.csv'), index=False)

        # Visualize comparison
        self._plot_methods_comparison(all_results)

    def _plot_methods_comparison(self, all_results):
        """Plot methods comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        methods = [r['method'] for r in all_results]

        # Accuracy
        axes[0, 0].bar(methods, [r['accuracy'] for r in all_results])
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=15)

        # Balanced Accuracy
        axes[0, 1].bar(methods, [r['balanced_accuracy'] for r in all_results])
        axes[0, 1].set_title('Balanced Accuracy')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=15)

        # F1-Macro
        axes[1, 0].bar(methods, [r['f1_macro'] for r in all_results])
        axes[1, 0].set_title('F1-Score (Macro)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=15)

        # Per-class F1
        x = np.arange(len(self.class_names))
        width = 0.35
        for i, result in enumerate(all_results):
            f1_scores = [result['f1_per_class'][cls] for cls in self.class_names]
            axes[1, 1].bar(x + i*width, f1_scores, width, label=result['method'])

        axes[1, 1].set_title('Per-Class F1-Score')
        axes[1, 1].set_xticks(x + width / 2)
        axes[1, 1].set_xticklabels(self.class_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'methods_comparison.png'))
        plt.close()
