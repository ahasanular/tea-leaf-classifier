#!/usr/bin/env python3
"""
Tea Leaf Disease Classification System
Modular, production-ready implementation with prototype-based explainable AI
"""

import sys
import torch
import warnings
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Suppress warnings
warnings.filterwarnings('ignore')

from default_config import Config
from dataset import DataModule
from models import TeaLeafModel
from trainer import PrototypeTrainer
from metrics import ModelEvaluator
from visualization.plots import VisualizationEngine
from dataset import TeaLeafDataSplitter


class TeaLeafClassificationSystem:
    """Main system class that orchestrates the entire pipeline"""

    def __init__(self, config_class=Config):
        self.config = config_class()
        self.data_module = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = None

    def setup(self):
        """Setup all components of the system"""
        print("=== Tea Leaf Disease Classification System ===")
        self.config.print_config()
        print("\n" + "=" * 50)

        # Setup data
        print("\n[1/5] Setting up data pipeline...")
        self.data_module = DataModule(self.config)
        self.data_module.prepare_data()

        # Setup model
        print("\n[2/5] Initializing model...")
        self.model = TeaLeafModel(
            num_classes=len(self.data_module.splits['known_classes']),
            config=self.config
        ).to(self.config.DEVICE)

        # Setup trainer
        print("\n[3/5] Setting up training components...")
        self.trainer = PrototypeTrainer(self.model, self.data_module, self.config)

        # Setup evaluator and visualizer
        print("\n[4/5] Setting up evaluation and visualization...")
        self.evaluator = ModelEvaluator(self.model, self.data_module, self.config)
        self.visualizer = VisualizationEngine(self.model, self.data_module, self.config)

        print("\n[5/5] System setup complete!")
        return self

    def train(self):
        """Train the model"""
        if self.trainer is None:
            self.setup()

        print("\n" + "=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)

        history, best_state = self.trainer.train()

        # Load best model for evaluation
        if best_state:
            self.model.backbone.load_state_dict(best_state['backbone'])
            self.model.head.load_state_dict(best_state['head'])

        return history, best_state

    def evaluate(self, tag="main"):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # Get validation predictions
        val_acc, val_logits, val_labels, val_paths = self.trainer.validate()
        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)

        print(f"Validation Accuracy: {val_acc:.3f}")

        # Generate all evaluation artifacts
        export_dir = self.config.EXPORT_DIR

        # 1. Confusion Matrix
        print("\n[1/6] Generating confusion matrix...")
        self.evaluator.compute_confusion_matrix(
            val_logits.numpy(), val_labels.numpy(),
            save_path=export_dir / f"{tag}_confusion_matrix.png"
        )

        # 2. Classification Report
        print("[2/6] Generating classification report...")
        class_report = self.evaluator.compute_classification_report(
            val_logits.numpy(), val_labels.numpy(),
            save_path=export_dir / f"{tag}_classification_report"
        )
        print("Class-wise performance:")
        print(class_report.round(3))

        # 3. Reliability Diagram
        print("[3/6] Generating reliability diagram...")
        ece = self.evaluator.reliability_diagram(
            val_logits.numpy(), val_labels.numpy(),
            save_path=export_dir / f"{tag}_reliability_diagram.png"
        )
        print(f"Expected Calibration Error: {ece:.4f}")

        # 4. OOD Analysis
        print("[4/6] Performing OOD analysis...")
        ood_metrics = self.evaluator.compute_ood_metrics(
            self.data_module.val_loader,
            self.data_module.ood_loader
        )

        if ood_metrics:
            print(f"OOD Detection - AUROC: {ood_metrics['auroc']:.3f}, "
                  f"FPR@95TPR: {ood_metrics['fpr_95']:.3f}")
            self.visualizer.plot_ood_analysis(
                ood_metrics,
                save_path=export_dir / f"{tag}_ood_analysis.png"
            )

        # 5. t-SNE Visualization
        print("[5/6] Generating t-SNE plot...")
        self.evaluator.generate_tsne_plot(
            self.data_module.val_loader,
            self.data_module.ood_loader,
            save_path=export_dir / f"{tag}_tsne_visualization.png"
        )

        # 6. Prototype Visualizations
        print("[6/6] Generating prototype visualizations...")
        self.visualizer.visualize_prototype_overlays(
            self.data_module.val_ds,
            save_path=export_dir / f"{tag}_prototype_overlays.png"
        )

        self.visualizer.export_prototype_tiles(
            self.data_module.train_ds,
            save_dir=export_dir / f"{tag}_prototype_tiles"
        )

        print(f"\nAll evaluation artifacts saved to: {export_dir}")

        return {
            'validation_accuracy': val_acc,
            'ece': ece,
            'ood_metrics': ood_metrics,
            'class_report': class_report
        }

    def run_ood_sweep(self):
        """Run OOD detection sweep across all classes"""
        if not self.config.RUN_OOD_SWEEP:
            print("OOD sweep is disabled in config. Set RUN_OOD_SWEEP=True to enable.")
            return

        print("\n" + "=" * 50)
        print("OOD DETECTION SWEEP")
        print("=" * 50)

        sweep_results = []
        known_classes = self.data_module.splits['known_classes']

        for unknown_class in known_classes:
            print(f"\nRunning OOD sweep with '{unknown_class}' as unknown...")

            # Create new data split with current class as unknown
            temp_splitter = TeaLeafDataSplitter(
                self.config.DATA_ROOT,
                unknown_class=unknown_class,
                val_ratio=self.config.VAL_RATIO,
                seed=self.config.SEED
            )
            temp_splits = temp_splitter.build_splits()

            # Skip if this would leave too few classes
            if len(temp_splits['known_classes']) < 2:
                print(f"  Skipping {unknown_class} - too few classes remaining")
                continue

            # Train and evaluate with this OOD setup
            temp_model = TeaLeafModel(
                num_classes=len(temp_splits['known_classes']),
                config=self.config
            ).to(self.config.DEVICE)

            temp_data_module = DataModule(self.config)
            temp_data_module.splits = temp_splits
            # Note: You'd need to recreate loaders here - simplified for example

            # For now, just record the configuration
            sweep_results.append({
                'unknown_class': unknown_class,
                'remaining_classes': len(temp_splits['known_classes']),
                'ood_samples': len(temp_splits['ood_files'])
            })

        # Save sweep summary
        import pandas as pd
        sweep_df = pd.DataFrame(sweep_results)
        sweep_path = self.config.EXPORT_DIR / "ood_sweep_summary.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nOOD sweep summary saved to: {sweep_path}")

        return sweep_results


def main():
    """Main execution function"""
    try:
        # Initialize the system
        system = TeaLeafClassificationSystem()

        # Setup and train
        system.setup()
        history, best_state = system.train()

        # Evaluate
        results = system.evaluate()

        # Optional: Run OOD sweep
        if system.config.RUN_OOD_SWEEP:
            system.run_ood_sweep()

        print("\n" + "=" * 50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Best Validation Accuracy: {results['validation_accuracy']:.3f}")

        if results['ood_metrics']:
            print(f"OOD Detection AUROC: {results['ood_metrics']['auroc']:.3f}")

        print(f"\nAll artifacts saved to: {system.config.EXPORT_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your dataset path and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
