import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_fscore_support)
from sklearn.manifold import TSNE
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics - COMPUTATION ONLY, NO PLOTTING"""

    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = config.DEVICE

    def compute_confusion_matrix(self, logits, true_labels):
        """Compute confusion matrix - returns raw data only"""
        pred_labels = logits.argmax(1)
        cm = confusion_matrix(true_labels, pred_labels,
                              labels=list(range(len(self.data_module.splits['known_classes']))))
        return cm

    def compute_classification_report(self, logits, true_labels):
        """Compute detailed classification report - with robust error handling"""
        pred_labels = logits.argmax(1)

        # Generate the report in dictionary form for data extraction
        report_dict = classification_report(
            true_labels, pred_labels,
            target_names=self.data_module.splits['known_classes'],
            digits=4,
            output_dict=True
        )

        # Convert to DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # --- Safely extract overall metrics ---
        def safe_float_extract(data_source, primary_key, secondary_key=None, default=0.0):
            """Safely extract a float value from nested dictionary structure."""
            try:
                if secondary_key is not None:
                    # For nested keys like report_dict['weighted avg']['precision']
                    value = data_source.get(primary_key, {}).get(secondary_key, default)
                else:
                    # For direct keys like report_dict['accuracy']
                    value = data_source.get(primary_key, default)
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        # Build robust metrics with safe extraction
        robust_metrics = {
            'accuracy': safe_float_extract(report_dict, 'accuracy'),
            'weighted_avg': {
                'precision': safe_float_extract(report_dict, 'weighted avg', 'precision'),
                'recall': safe_float_extract(report_dict, 'weighted avg', 'recall'),
                'f1_score': safe_float_extract(report_dict, 'weighted avg', 'f1-score')
            },
            'macro_avg': {
                'precision': safe_float_extract(report_dict, 'macro avg', 'precision'),
                'recall': safe_float_extract(report_dict, 'macro avg', 'recall'),
                'f1_score': safe_float_extract(report_dict, 'macro avg', 'f1-score')
            }
        }

        # Store for later use
        report_df.attrs['robust_metrics'] = robust_metrics

        # Validate that we have numeric values
        print(f"[DEBUG] Accuracy type: {type(robust_metrics['accuracy'])}, value: {robust_metrics['accuracy']}")
        print(
            f"[DEBUG] Weighted precision type: {type(robust_metrics['weighted_avg']['precision'])}, value: {robust_metrics['weighted_avg']['precision']}")

        return report_df

    def compute_reliability_metrics(self, logits, true_labels, n_bins=12):
        """Compute Expected Calibration Error and related data - NO PLOTTING"""
        probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracy = (predictions == np.array(true_labels)).astype(np.float32)

        # Bin the confidence scores
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(confidence, bin_boundaries) - 1

        ece = 0.0
        bin_data = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.any():
                bin_accuracy = accuracy[mask].mean()
                bin_confidence = confidence[mask].mean()
                bin_weight = mask.mean()

                ece += bin_weight * abs(bin_accuracy - bin_confidence)
                bin_data.append({
                    'bin_index': bin_idx,
                    'accuracy': float(bin_accuracy),
                    'confidence': float(bin_confidence),
                    'count': int(mask.sum())
                })

        return {
            'ece': float(ece),
            'bin_data': bin_data,
            'confidence_scores': confidence.tolist(),
            'accuracy_values': accuracy.tolist()
        }

    def compute_ood_metrics(self, val_loader, ood_loader):
        """Compute Out-of-Distribution detection metrics - NO PLOTTING"""
        if ood_loader is None:
            return None

        # Get energy scores
        energy_in = self._get_energy_scores(val_loader)
        energy_out = self._get_energy_scores(ood_loader)

        # Create binary labels (0: in-distribution, 1: out-of-distribution)
        y_true = np.concatenate([
            np.zeros_like(energy_in),
            np.ones_like(energy_out)
        ])
        scores = np.concatenate([energy_in, energy_out])

        # Compute AUROC
        auroc = roc_auc_score(y_true, scores)

        # Compute FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fpr_95 = self._get_fpr_at_tpr(fpr, tpr, target_tpr=0.95)

        return {
            'auroc': auroc,
            'fpr_95': fpr_95,
            'energy_in': energy_in.tolist(),
            'energy_out': energy_out.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

    def compute_tsne_embeddings(self, data_loader, ood_loader=None):
        """Compute t-SNE embeddings - returns data only, NO PLOTTING"""
        self.model.eval()

        # Collect features
        features, labels = self._collect_features(data_loader, label_offset=0)

        if ood_loader is not None:
            features_ood, labels_ood = self._collect_features(
                ood_loader,
                label_offset=len(self.data_module.splits['known_classes'])
            )
            features = np.vstack([features, features_ood])
            labels = np.concatenate([labels, labels_ood])
            class_names = self.data_module.splits['known_classes'] + [
                f"OOD:{self.data_module.splits['resolved_unknown']}"
            ]
        else:
            class_names = self.data_module.splits['known_classes']

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(features) - 1),
            learning_rate='auto',
            init='pca',
            random_state=self.config.SEED,
            max_iter=1000
        )

        embeddings = tsne.fit_transform(features[:self.config.TSNE_MAX_ITEMS])
        labels_tsne = labels[:self.config.TSNE_MAX_ITEMS]

        return {
            'embeddings': embeddings.tolist(),
            'labels': labels_tsne.tolist(),
            'class_names': class_names
        }

    def _get_energy_scores(self, data_loader):
        """Compute energy scores for OOD detection"""
        self.model.eval()
        energy_scores = []

        with torch.no_grad():
            for x, _, _ in data_loader:
                x = x.to(self.device)
                logits, _, _ = self.model(x)
                # Energy score: -log(sum(exp(logits)))
                energy = -torch.logsumexp(logits, dim=1)
                energy_scores.append(energy.cpu().numpy())

        return np.concatenate(energy_scores)

    def _get_fpr_at_tpr(self, fpr, tpr, target_tpr=0.95):
        """Get False Positive Rate at specific True Positive Rate"""
        try:
            idx = np.where(tpr >= target_tpr)[0][0]
            return fpr[idx]
        except IndexError:
            return 1.0  # Worst case

    def _collect_features(self, data_loader, label_offset=0, max_samples=None):
        """Collect features from the backbone"""
        features = []
        labels = []
        count = 0

        with torch.no_grad():
            for x, y, _ in data_loader:
                x = x.to(self.device)
                # Get features before the prototype head
                backbone_features = self.model.backbone(x)
                # Global average pooling
                pooled_features = F.adaptive_avg_pool2d(backbone_features, 1)
                flattened_features = pooled_features.flatten(1).cpu().numpy()

                features.append(flattened_features)

                if isinstance(y, torch.Tensor) and y.ndim > 0:
                    batch_labels = (y.numpy() + label_offset).tolist()
                else:
                    batch_labels = [label_offset] * flattened_features.shape[0]

                labels.extend(batch_labels)
                count += flattened_features.shape[0]

                if max_samples and count >= max_samples:
                    break

        return np.concatenate(features), np.array(labels)