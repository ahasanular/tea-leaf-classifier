#!/usr/bin/env python3
"""
Tea Leaf Disease Classification - Plot Generator
Fixed version for your training_results.json structure
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from default_config import Config


class TeaLeafPlotGenerator:
    """Generate comprehensive plots from training results JSON"""

    def __init__(self, config):
        self.config = config()
        input_result = Path(self.config.EXPORT_DIR) / 'training_results.json'
        self.results_json_path = input_result

        if not self.results_json_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_result}")

        # Load results
        with open(self.results_json_path, 'r') as f:
            self.results = json.load(f)

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

        ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
        ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # Plot 2: Training Loss
        train_loss = [entry.get('train_loss', 0) for entry in history]
        ax2.plot(epochs, train_loss, 'g-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning Rate
        if 'learning_rate' in history[0]:
            lr_values = [entry.get('learning_rate', 0) for entry in history]
            ax3.plot(epochs, lr_values, 'purple', linewidth=2, label='Learning Rate', marker='o', markersize=4)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')

        # Plot 4: Accuracy Difference (Val - Train)
        acc_diff = [val - train for train, val in zip(train_acc, val_acc)]
        ax4.plot(epochs, acc_diff, 'orange', linewidth=2, label='Val - Train Accuracy', marker='o', markersize=4)
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
            # Get confusion matrix from test_results
            if 'test_results' in self.results and 'confusion_matrix' in self.results['test_results']:
                cm = np.array(self.results['test_results']['confusion_matrix'])
            else:
                print("⚠️ No confusion matrix found in results")
                return

            class_names = self.results['class_information']['known_classes']

            plt.figure(figsize=(10, 8))

            # Create heatmap with better formatting
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'})

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
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
            bars1 = ax1.bar(range(len(class_names)), train_counts, color='skyblue', edgecolor='black', alpha=0.7)
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
            bars2 = ax2.bar(range(len(class_names)), val_counts, color='lightcoral', edgecolor='black', alpha=0.7)
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
                print("⚠️ No classification report found in results")
                return

            class_report = self.results['test_results']['classification_report']
            class_names = self.results['class_information']['known_classes']

            # Extract metrics for each class (skip 'accuracy', 'macro avg', 'weighted avg')
            precision = []
            recall = []
            f1_scores = []
            supports = []

            valid_classes = []
            for class_name in class_names:
                if class_name in class_report:
                    class_data = class_report[class_name]
                    precision.append(class_data.get('precision', 0))
                    recall.append(class_data.get('recall', 0))
                    f1_scores.append(class_data.get('f1-score', 0))
                    supports.append(class_data.get('support', 0))
                    valid_classes.append(class_name)

            if not valid_classes:
                print("⚠️ No valid class data found in classification report")
                return

            x = np.arange(len(valid_classes))
            width = 0.25

            fig, ax = plt.subplots(figsize=(12, 6))

            bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.7, edgecolor='black')
            bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7, edgecolor='black')

            ax.set_xlabel('Classes')
            ax.set_ylabel('Scores')
            ax.set_title('Per-Class Performance Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_classes, rotation=45, ha='right')
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
            import traceback
            traceback.print_exc()

    def plot_ood_analysis(self):
        """Plot OOD analysis if data is available"""
        try:
            # Check if OOD data exists
            class_info = self.results.get('class_information', {})
            if 'ood' in class_info.get('class_distribution', {}) and class_info['class_distribution']['ood'] > 0:
                # Create a simple OOD information plot
                fig, ax = plt.subplots(figsize=(8, 6))

                categories = ['Known Classes', 'OOD (Helopeltis)']
                counts = [
                    sum(self.results['class_information']['class_distribution']['val']),
                    self.results['class_information']['class_distribution']['ood']
                ]

                bars = ax.bar(categories, counts, color=['skyblue', 'lightcoral'], alpha=0.7, edgecolor='black')
                ax.set_ylabel('Number of Samples')
                ax.set_title('In-Distribution vs Out-of-Distribution Samples')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(self.output_dir / 'ood_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ OOD analysis plot saved")
            else:
                print("ℹ️ No OOD data available for plotting")

        except Exception as e:
            print(f"⚠️ Could not plot OOD analysis: {e}")

    def generate_performance_summary(self):
        """Generate a comprehensive performance summary"""
        try:
            if 'test_results' not in self.results:
                print("⚠️ No test results found for performance summary")
                return

            test_results = self.results['test_results']
            class_report = test_results.get('classification_report', {})

            # Create summary table
            summary_data = []
            class_names = self.results['class_information']['known_classes']

            for class_name in class_names:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    summary_data.append({
                        'Class': class_name,
                        'Precision': f"{metrics.get('precision', 0):.3f}",
                        'Recall': f"{metrics.get('recall', 0):.3f}",
                        'F1-Score': f"{metrics.get('f1-score', 0):.3f}",
                        'Support': int(metrics.get('support', 0))
                    })

            # Add overall accuracy
            accuracy = test_results.get('accuracy', 0)

            # Create figure for summary table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')

            if summary_data:
                # Create table
                df = pd.DataFrame(summary_data)
                table = ax.table(cellText=df.values,
                                 colLabels=df.columns,
                                 cellLoc='center',
                                 loc='center',
                                 bbox=[0, 0, 1, 0.9])

                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)

                # Add title with accuracy
                plt.suptitle(f'Performance Summary - Overall Accuracy: {accuracy:.3f}', fontsize=14, y=0.95)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Performance summary plot saved")

        except Exception as e:
            print(f"⚠️ Could not generate performance summary: {e}")

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
        self.generate_performance_summary()

        print("\n" + "=" * 50)
        print("🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
        print(f"📁 Location: {self.output_dir}")
        print("=" * 50)


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate plots from tea leaf training results')
    parser.add_argument('--results', '-r', default='./output2/training_results.json',
                        help='Path to training_results.json file')
    parser.add_argument('--output', '-o',
                        help='Output directory for plots (default: same as results file)')

    args = parser.parse_args()

    try:
        # Initialize plot generator
        plotter = TeaLeafPlotGenerator(config=Config)

        # Generate all plots
        plotter.generate_all_plots()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()