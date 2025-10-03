import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class TeaLeafResultsLogger:
    """
    Comprehensive results logger for Tea Leaf Disease Classification
    Stores all analytics in memory and saves to JSON at the end
    """

    def __init__(self, config, export_dir):
        self.config = config
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results structure - everything stored in memory
        self.results = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'project': 'TeaLeafBD_Classification',
                'version': '1.0',
                'device_used': config.DEVICE,
                'completed': False
            },
            'training_config': self._extract_config(),
            'training_history': [],
            'class_information': {},
            'final_model_performance': {},
            'test_results': {},
            'paths_and_checkpoints': {}
        }

        # Track best model state
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def _extract_config(self):
        """Extract configuration in CRC-HGD format"""
        return {
            'data': {
                'dataset_path': self.config.DATA_ROOT,
                'image_size': self.config.IMG_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'val_ratio': self.config.VAL_RATIO,
                'unknown_class': self.config.UNKNOWN_CLASS_NAME
            },
            'model': {
                'backbone': self.config.BACKBONE_NAME,
                'prototype_dim': self.config.PROTOTYPE_DIM,
                'protos_per_class': self.config.PROTOS_PER_CLASS,
                'img_size': self.config.IMG_SIZE
            },
            'training': {
                'epochs': self.config.EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'weight_decay': self.config.WEIGHT_DECAY,
                'warmup_epochs': self.config.WARMUP_EPOCHS,
                'grad_clip': self.config.GRAD_CLIP
            },
            'experiment': {
                'seed': self.config.SEED,
                'output_dir': str(self.config.EXPORT_DIR),
                'save_checkpoints': True
            }
        }

    def log_class_information(self, data_module):
        """Log class distribution and information"""
        splits = data_module.splits
        known_classes = splits['known_classes']

        # Calculate class distribution
        train_dist = [0] * len(known_classes)
        val_dist = [0] * len(known_classes)

        for label in splits['train_labels']:
            train_dist[splits['id_remap'][label]] += 1
        for label in splits['val_labels']:
            val_dist[splits['id_remap'][label]] += 1

        self.results['class_information'] = {
            'known_classes': known_classes,
            'unknown_class': splits['resolved_unknown'],
            'class_distribution': {
                'train': train_dist,
                'val': val_dist,
                'ood': len(splits['ood_files']) if splits['ood_files'] else 0
            },
            'total_samples': {
                'train': len(splits['train_files']),
                'val': len(splits['val_files']),
                'ood': len(splits['ood_files']) if splits['ood_files'] else 0
            }
        }

    def log_training_epoch(self, epoch, train_loss, train_acc, val_acc, lr, prototype_push=False):
        """
        Log comprehensive epoch analytics - stored in memory
        """
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'learning_rate': float(lr),
            'prototype_push_epoch': prototype_push
        }

        # Update best model tracking
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch

        # Store in memory
        self.results['training_history'].append(epoch_data)

    def log_final_metrics(self, test_accuracy, classification_report,
                          confusion_matrix, true_labels, predictions, ood_metrics=None):
        """Log final test metrics"""

        # Convert classification report to dictionary if it's a string
        if isinstance(classification_report, str):
            report_dict = self._parse_classification_report(classification_report)
        else:
            report_dict = classification_report

        self.results['test_results'] = {
            'accuracy': float(test_accuracy),
            'classification_report': report_dict,
            'confusion_matrix': confusion_matrix.tolist() if hasattr(confusion_matrix, 'tolist') else confusion_matrix,
            'true_labels': true_labels if isinstance(true_labels, list) else true_labels.tolist(),
            'predictions': predictions if isinstance(predictions, list) else predictions.tolist(),
            'ood_detection_metrics': ood_metrics
        }

    def log_final_model_performance(self, overall_metrics, per_class_metrics,
                                    confusion_matrix, ood_metrics=None):
        """Log final model performance summary"""
        self.results['final_model_performance'] = {
            'best_epoch': self.best_epoch,
            'best_val_accuracy': float(self.best_val_accuracy),
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix.tolist() if hasattr(confusion_matrix, 'tolist') else confusion_matrix,
            'ood_detection_metrics': ood_metrics,
            'training_duration_epochs': len(self.results['training_history'])
        }

    def log_paths(self, model_paths):
        """Log paths to saved models"""
        self.results['paths_and_checkpoints'] = {
            'best_model': model_paths.get('best_model', ''),
            'final_model': model_paths.get('final_model', ''),
            'export_dir': str(self.export_dir)
        }

    def _parse_classification_report(self, report_str):
        """Parse sklearn classification report string into dictionary"""
        lines = report_str.split('\n')
        report_dict = {}

        for line in lines:
            if line.strip() and not line.startswith('---'):
                parts = line.split()
                if len(parts) >= 5 and parts[0] != 'accuracy':
                    class_name = parts[0]
                    report_dict[class_name] = {
                        'precision': float(parts[1]),
                        'recall': float(parts[2]),
                        'f1_score': float(parts[3]),
                        'support': int(parts[4])
                    }
                elif parts[0] == 'accuracy':
                    report_dict['accuracy'] = float(parts[1])
                    report_dict['macro_avg'] = {
                        'precision': float(parts[5]),
                        'recall': float(parts[6]),
                        'f1_score': float(parts[7])
                    }

        return report_dict

    def save_results(self, filename="training_results.json"):
        """
        Save all collected analytics to JSON file - called only at the end
        """
        self.results['experiment_metadata']['completed'] = True
        self.results['experiment_metadata']['end_timestamp'] = datetime.now().isoformat()

        filepath = self.export_dir / filename

        # Convert any tensors/numpy arrays to Python lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist() if obj.device != 'cpu' else obj.numpy().tolist()
            elif isinstance(obj, (list, dict, str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Recursively convert the results dictionary
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_for_json(d)

        json_ready_results = convert_dict(self.results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_ready_results, f, indent=2, ensure_ascii=False)

        print(f"\n[Logger] All analytics saved to {filepath}")
        print(f"[Logger] Training history: {len(self.results['training_history'])} epochs")
        print(f"[Logger] Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")

        return filepath

    def get_training_history(self):
        """Get training history for plotting"""
        return self.results['training_history']

    def get_final_metrics(self):
        """Get final metrics for analysis"""
        return self.results.get('final_model_performance', {})
