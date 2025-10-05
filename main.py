#!/usr/bin/env python3
"""
Tea Leaf Disease Classification System
Clean implementation: Only trains, saves models, collects analytics, saves JSON
NO PLOTTING during main execution
"""

import os
import sys
import torch
import warnings
import numpy as np
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
from utils.results_logger import TeaLeafResultsLogger
from visualizer import PrototypeOverlayVisualizer, OODRocVisualizer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_fscore_support)
import pandas as pd


class TeaLeafClassificationSystem:
    """Main system class - ONLY trains, saves models, collects analytics"""

    def __init__(self, config_class=Config):
        self.config = config_class()
        self.data_module = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.results_logger = None

    def setup(self):
        """Setup all components of the system"""
        print("=== Tea Leaf Disease Classification System ===")
        print(f"Device: {self.config.DEVICE}")
        print(f"Output directory: {self.config.EXPORT_DIR}")

        # Check dataset existence
        if not Path(self.config.DATA_ROOT).exists():
            raise FileNotFoundError(f"Dataset not found at {self.config.DATA_ROOT}")

        # Setup data
        print("\n[1/4] Setting up data pipeline...")
        self.data_module = DataModule(self.config)
        self.data_module.prepare_data()

        # Setup model
        print("[2/4] Initializing model...")
        self.model = TeaLeafModel(
            num_classes=len(self.data_module.splits['known_classes']),
            config=self.config
        ).to(self.config.DEVICE)

        # Setup results logger
        print("[3/4] Initializing analytics logger...")
        self.results_logger = TeaLeafResultsLogger(self.config, self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME)
        self.results_logger.log_class_information(self.data_module)

        # Setup trainer
        print("[4/4] Setting up training components...")
        self.trainer = PrototypeTrainer(self.model, self.data_module, self.config)

        self.trainer.results_logger = self.results_logger

        print("\n✅ System setup complete!")
        return self

    def train(self):
        """Train the model with comprehensive analytics logging"""
        if self.trainer is None:
            self.setup()

        print("\n" + "=" * 50)
        print("STARTING TRAINING WITH ANALYTICS LOGGING")
        print("=" * 50)
        print(f"Training for {self.config.EPOCHS} epochs")
        print(f"Classes: {self.data_module.splits['known_classes']}")

        # Train the model
        history, best_state = self.trainer.train()

        # Load best model for final evaluation
        if best_state:
            self.model.backbone.load_state_dict(best_state['backbone'])
            self.model.head.load_state_dict(best_state['head'])
            print(f"✅ Loaded best model for final evaluation")

        return history, best_state

    def evaluate(self):
        """Evaluate model and collect metrics - NO PLOTTING"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 50)
        print("FINAL MODEL EVALUATION - COLLECTING METRICS")
        print("=" * 50)

        # Setup evaluator for metrics collection only
        self.evaluator = ModelEvaluator(self.model, self.data_module, self.config)

        # Get validation predictions
        val_acc, val_logits, val_labels, val_paths = self.trainer.validate()
        val_logits_tensor = torch.cat(val_logits)
        val_labels_tensor = torch.cat(val_labels)

        print(f"Validation Accuracy: {val_acc:.4f}")

        # Compute metrics WITHOUT plotting
        print("\n[1/6] Computing confusion matrix...")
        cm = self.evaluator.compute_confusion_matrix(
            val_logits_tensor.numpy(),
            val_labels_tensor.numpy(),
            # save_path=None  # No plotting!
        )

        print("[2/6] Computing classification report...")
        class_report = self.evaluator.compute_classification_report(
            val_logits_tensor.numpy(),
            val_labels_tensor.numpy(),
        )

        print("[3/6] Computing reliability metrics...")
        reliability_metrics = self.evaluator.compute_reliability_metrics(
            val_logits_tensor.numpy(),
            val_labels_tensor.numpy(),
            # save_path=None  # No plotting!
        )

        print("[4/6] Computing OOD metrics...")
        ood_metrics = self.evaluator.compute_ood_metrics(
            self.data_module.val_loader,
            self.data_module.ood_loader
        )

        print("[5/6] Generating OOD metrics plots...")
        plotter = OODRocVisualizer(self.config)
        plotter.generate(result=ood_metrics)

        print("[6/6] Generating prototype overlay...")
        visualizer = PrototypeOverlayVisualizer(self.model, self.config)
        visualizer.visualize_prototypes(self.data_module.val_ds, samples=6)

        overall_metrics = {
            'accuracy': val_acc,
            'precision': float(class_report['weighted avg']['precision']),
            'recall': float(class_report['weighted avg']['recall']),
            'f1_score': float(class_report['weighted avg']['f1-score']),
            'ece': reliability_metrics.get('ece')
        }
        # Log final results to analytics
        self.results_logger.log_final_metrics(
            overall_metrics=overall_metrics,
            test_accuracy=val_acc,
            classification_report=class_report,
            confusion_matrix=cm,
            true_labels=val_labels_tensor.numpy(),
            predictions=val_logits_tensor.argmax(1).numpy(),
            ood_metrics=ood_metrics
        )

        # Log model paths
        model_paths = {
            'best_model': str(self.config.EXPORT_DIR/ self.config.UNKNOWN_CLASS_NAME / "best_model.pth"),
            'final_model': str(self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / "final_model.pth")
        }
        self.results_logger.log_paths(model_paths)

        # Save final model
        torch.save({
            'backbone': self.model.backbone.state_dict(),
            'head': self.model.head.state_dict(),
            'known_classes': self.data_module.splits['known_classes'],
            'config': self.config.__dict__
        }, self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / "final_model.pth")

        # Save all analytics to JSON
        results_file = self.results_logger.save_results("training_results.json")

        print(f"\n✅ Evaluation complete - All metrics collected!")
        print(f"📊 Validation Accuracy: {val_acc:.4f}")
        print(f"💾 Analytics JSON: {results_file}")
        print(f"💾 Final model: {self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / 'final_model.pth'}")

        return {
            'validation_accuracy': val_acc,
            'ece': reliability_metrics['ece'],
            'ood_metrics': ood_metrics,
            'class_report': class_report,
            'results_file': results_file,
            'known_classes': self.data_module.splits['known_classes'],
        }

    def run_complete_pipeline(self):
        """Run complete training and evaluation pipeline"""
        try:
            # Setup
            self.setup()

            # Train
            print("\n🚀 Starting training...")
            history, best_state = self.train()

            # Evaluate (collect metrics only)
            print("\n📈 Collecting final metrics...")
            results = self.evaluate()

            print("\n" + "=" * 50)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📈 Best Validation Accuracy: {results['validation_accuracy']:.4f}")

            if results['ood_metrics']:
                print(f"🚨 OOD Detection AUROC: {results['ood_metrics']['auroc']:.4f}")

            print(f"💾 Analytics JSON: {results['results_file']}")
            # print(f"🔧 Next: Run 'python generate_plots.py' to generate all visualizations")

            from generate_plots import TeaLeafPlotGenerator
            plotter = TeaLeafPlotGenerator(config=self.config)
            plotter.generate_all_plots()

            known_classes = results['known_classes']
            if self.config.RUN_OOD_SWEEP:
                for known_class in known_classes:
                    print(f"\n 🚀 Running with new unknows class {known_class}...")
                    self.config.UNKNOWN_CLASS_NAME = known_class
                    # self.config.EXPORT_DIR = Path(f"./output-ood-{known_class}")
                    self.setup()

                    # Train
                    print("\n🚀 Starting training...")
                    history, best_state = self.train()

                    print("\n📈 Collecting final metrics...")
                    results = self.evaluate()

                    print("\n" + "=" * 50)
                    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                    print("=" * 50)
                    print(f"📈 Best Validation Accuracy: {results['validation_accuracy']:.4f}")

                    if results['ood_metrics']:
                        print(f"🚨 OOD Detection AUROC: {results['ood_metrics']['auroc']:.4f}")

                    from generate_plots import TeaLeafPlotGenerator
                    plotter = TeaLeafPlotGenerator(config=self.config)
                    plotter.generate_all_plots()

                from visualizer import OODComparisonTable
                table = OODComparisonTable(self.config)
                table.collect_results()
                # table.show_summary()
                table.save_csv()
            return results

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    """Main execution function"""
    try:
        system = TeaLeafClassificationSystem()
        results = system.run_complete_pipeline()

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
