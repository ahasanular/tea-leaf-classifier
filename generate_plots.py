#!/usr/bin/env python3
"""
Tea Leaf Disease Classification - Plot Generator
Standalone script to generate all visualizations from training_results.json
Usage: python generate_plots.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys


class TeaLeafPlotGenerator:
    """Generate comprehensive plots from training results JSON"""

    def __init__(self, results_json_path, output_dir=None):
        """
        Initialize plot generator

        Args:
            results_json_path: Path to training_results.json
            output_dir: Directory to save plots (default: same as JSON file)
        """
        self.results_json_path = Path(results_json_path)

        if not self.results_json_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_json_path}")

        # Load results
        with open(self.results_json_path, 'r') as f:
            self.results = json.load(f)

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_json_path.parent / "plots"

        self.output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        print(f"📊 Loaded results from: {self.results_json_path}")
        print(f"💾 Saving plots to: {self.output_dir}")

    def plot_training_curves(self):
        """Plot training and validation accuracy/loss curves"""
        if 'training_history' not in self.results or not self.results['training_history']:
            print("⚠️ No training history found in results")
            return

        history = self.results['training_history']
        epochs = [entry['epoch'] for entry in history]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training and Validation Accuracy
        train_acc = [entry.get('train_accuracy', 0) for entry in history]
        val_acc = [entry.get('val_accuracy', 0) for entry in history]

        ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
        ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # Plot 2: Training Loss
        train_loss = [entry.get('train_loss', 0) for entry in history]
        ax2.plot(epochs, train_loss, 'g-', linewidth=2, label='Training Loss', marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning Rate
        if 'learning_rate' in history[0]:
            lr_values = [entry.get('learning_rate', 0) for entry in history]
            ax3.plot(epochs, lr_values, 'purple', linewidth=2, label='Learning Rate', marker='o', markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')

        # Plot 4: Accuracy Difference (Val - Train)
        acc_diff = [val - train for train, val in zip(train_acc, val_acc)]
        ax4.plot(epochs, acc_diff, 'orange', linewidth=2, label='Val - Train Accuracy', marker='o', markersize=3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference')
        ax4.set_title('Validation - Training Accuracy Gap')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Training curves plot saved")

    def plot_confusion_matrix(self):
        """Plot confusion matrix from final metrics"""
        try:
            # Try to get confusion matrix from different possible locations
            cm = None
            if 'final_model_performance' in self.results and 'confusion_matrix' in self.results[
                'final_model_performance']:
                cm = np.array(self.results['final_model_performance']['confusion_matrix'])
            elif 'test_results' in self.results and 'confusion_matrix' in self.results['test_results']:
                cm = np.array(self.results['test_results']['confusion_matrix'])

            if cm is None:
                print("⚠️ No confusion matrix found in results")
                return

            class_names = self.results['class_information']['known_classes']

            plt.figure(figsize=(10, 8))

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Percentage'})

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Normalized Confusion Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Confusion matrix plot saved")

        except Exception as e:
            print(f"⚠️ Could not plot confusion matrix: {e}")

    def plot_class_distribution(self):
        """Plot class distribution in training and validation sets"""
        if 'class_information' not in self.results:
            print("⚠️ No class information found")
            return

        class_info = self.results['class_information']
        class_names = class_info['known_classes']

        if 'class_distribution' not in class_info:
            return

        dist = class_info['class_distribution']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Training distribution
        if 'train' in dist:
            train_counts = dist['train']
            bars1 = ax1.bar(range(len(class_names)), train_counts, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Number of Images')
            ax1.set_title('Training Set Class Distribution')
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

        # Validation distribution
        if 'val' in dist:
            val_counts = dist['val']
            bars2 = ax2.bar(range(len(class_names)), val_counts, color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Number of Images')
            ax2.set_title('Validation Set Class Distribution')
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Class distribution plot saved")

    def plot_per_class_metrics(self):
        """Plot per-class precision, recall, and F1-score"""
        try:
            if 'test_results' not in self.results or 'classification_report' not in self.results['test_results']:
                print("⚠️ No classification report found")
                return

            report = self.results['test_results']['classification_report']
            class_names = self.results['class_information']['known_classes']

            # Extract metrics for each class
            precision = []
            recall = []
            f1_scores = []

            for class_name in class_names:
                if class_name in report:
                    precision.append(report[class_name].get('precision', 0))
                    recall.append(report[class_name].get('recall', 0))
                    f1_scores.append(report[class_name].get('f1_score', 0))

            x = np.arange(len(class_names))
            width = 0.25

            fig, ax = plt.subplots(figsize=(12, 6))

            bars1 = ax.bar(x - width, precision, width, label='Precision', edgecolor='black')
            bars2 = ax.bar(x, recall, width, label='Recall', edgecolor='black')
            bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', edgecolor='black')

            ax.set_xlabel('Classes')
            ax.set_ylabel('Scores')
            ax.set_title('Per-Class Performance Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)

            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Per-class metrics plot saved")

        except Exception as e:
            print(f"⚠️ Could not plot per-class metrics: {e}")

    def plot_ood_analysis(self):
        """Plot OOD detection results if available"""
        try:
            ood_metrics = None

            # Try different locations for OOD metrics
            if 'test_results' in self.results and 'ood_detection_metrics' in self.results['test_results']:
                ood_metrics = self.results['test_results']['ood_detection_metrics']
            elif 'final_model_performance' in self.results and 'ood_detection_metrics' in self.results[
                'final_model_performance']:
                ood_metrics = self.results['final_model_performance']['ood_detection_metrics']

            if not ood_metrics:
                print("ℹ️ No OOD metrics found")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ROC Curve
            if 'fpr' in ood_metrics and 'tpr' in ood_metrics:
                ax1.plot(ood_metrics['fpr'], ood_metrics['tpr'], linewidth=2,
                         label=f'ROC Curve (AUC = {ood_metrics.get("auroc", "N/A"):.3f})')
                ax1.plot([0, 1], [0, 1], '--', color='gray', label='Random Classifier')
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate')
                ax1.set_title('OOD Detection ROC Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Energy distributions
            if 'energy_in' in ood_metrics and 'energy_out' in ood_metrics:
                ax2.hist(ood_metrics['energy_in'], bins=30, alpha=0.7,
                         label='In-distribution', density=True, color='blue')
                ax2.hist(ood_metrics['energy_out'], bins=30, alpha=0.7,
                         label='Out-of-distribution', density=True, color='red')
                ax2.set_xlabel('Energy Score')
                ax2.set_ylabel('Density')
                ax2.set_title('Energy Score Distributions\n(Higher = More OOD)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'ood_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ OOD analysis plot saved")

        except Exception as e:
            print(f"⚠️ Could not plot OOD analysis: {e}")

    def generate_summary_report(self):
        """Generate a text summary of key results"""
        summary_path = self.output_dir / 'experiment_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("TEA LEAF DISEASE CLASSIFICATION - EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Basic info
            if 'experiment_metadata' in self.results:
                meta = self.results['experiment_metadata']
                f.write(f"Project: {meta.get('project', 'N/A')}\n")
                f.write(f"Timestamp: {meta.get('timestamp', 'N/A')}\n")
                f.write(f"Device: {meta.get('device_used', 'N/A')}\n\n")

            # Class information
            if 'class_information' in self.results:
                class_info = self.results['class_information']
                f.write(f"Known Classes: {', '.join(class_info.get('known_classes', []))}\n")
                f.write(f"Unknown Class: {class_info.get('unknown_class', 'None')}\n\n")

            # Training results
            if 'training_history' in self.results and self.results['training_history']:
                history = self.results['training_history']
                final_epoch = history[-1]
                best_epoch = max(history, key=lambda x: x.get('val_accuracy', 0))

                f.write("TRAINING RESULTS:\n")
                f.write(f"Total Epochs: {len(history)}\n")
                f.write(f"Final Training Accuracy: {final_epoch.get('train_accuracy', 0):.4f}\n")
                f.write(f"Final Validation Accuracy: {final_epoch.get('val_accuracy', 0):.4f}\n")
                f.write(
                    f"Best Validation Accuracy: {best_epoch.get('val_accuracy', 0):.4f} (Epoch {best_epoch.get('epoch', 'N/A')})\n\n")

            # Final performance
            if 'final_model_performance' in self.results:
                perf = self.results['final_model_performance']
                f.write("FINAL PERFORMANCE:\n")
                if 'overall_metrics' in perf:
                    metrics = perf['overall_metrics']
                    f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"F1-Score: {metrics.get('f1_score', 0):.4f}\n")

            # OOD Results
            ood_metrics = None
            if 'test_results' in self.results and 'ood_detection_metrics' in self.results['test_results']:
                ood_metrics = self.results['test_results']['ood_detection_metrics']
            elif 'final_model_performance' in self.results and 'ood_detection_metrics' in self.results[
                'final_model_performance']:
                ood_metrics = self.results['final_model_performance']['ood_detection_metrics']

            if ood_metrics:
                f.write("\nOOD DETECTION:\n")
                f.write(f"AUROC: {ood_metrics.get('auroc', 0):.4f}\n")
                f.write(f"FPR@95TPR: {ood_metrics.get('fpr_95', 0):.4f}\n")

        print(f"✅ Experiment summary saved to {summary_path}")

    def generate_all_plots(self):
        """Generate all available plots"""
        print("\n" + "=" * 50)
        print("GENERATING ALL PLOTS FROM TRAINING RESULTS")
        print("=" * 50)

        self.plot_training_curves()
        self.plot_confusion_matrix()
        self.plot_class_distribution()
        self.plot_per_class_metrics()
        self.plot_ood_analysis()
        self.generate_summary_report()

        print("\n" + "=" * 50)
        print("🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
        print(f"📁 Location: {self.output_dir}")
        print("=" * 50)


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate plots from tea leaf training results')
    parser.add_argument('--results', '-r', default='./output/training_results.json',
                        help='Path to training_results.json file')
    parser.add_argument('--output', '-o',
                        help='Output directory for plots (default: same as results file)')

    args = parser.parse_args()

    try:
        # Initialize plot generator
        plotter = TeaLeafPlotGenerator(args.results, args.output)

        # Generate all plots
        plotter.generate_all_plots()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()