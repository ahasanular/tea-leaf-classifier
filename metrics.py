import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)


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
        """Compute detailed classification report - returns DICTIONARY, not string"""
        pred_labels = logits.argmax(1)

        # Get report as dictionary (not string)
        report_dict = classification_report(
            true_labels, pred_labels,
            target_names=self.data_module.splits['known_classes'],
            digits=4,
            output_dict=True  # This ensures we get a dictionary, not string
        )

        return report_dict

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
            'auroc': float(auroc),
            'fpr_95': float(fpr_95),
            'energy_in': energy_in.tolist(),
            'energy_out': energy_out.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
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