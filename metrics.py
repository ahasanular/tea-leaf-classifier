import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_fscore_support)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""

    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = config.DEVICE

    def compute_confusion_matrix(self, logits, true_labels, save_path=None):
        """Compute and plot confusion matrix"""
        pred_labels = logits.argmax(1)
        cm = confusion_matrix(true_labels, pred_labels,
                              labels=list(range(len(self.data_module.splits['known_classes']))))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.data_module.splits['known_classes'],
                    yticklabels=self.data_module.splits['known_classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return cm

    def compute_classification_report(self, logits, true_labels, save_path=None):
        """Compute detailed classification report"""
        pred_labels = logits.argmax(1)
        report = classification_report(
            true_labels, pred_labels,
            target_names=self.data_module.splits['known_classes'],
            digits=4,
            output_dict=True
        )

        # Convert to DataFrame for better handling
        report_df = pd.DataFrame(report).transpose()

        if save_path:
            # Save detailed report
            with open(save_path.with_suffix('.txt'), 'w') as f:
                f.write(classification_report(
                    true_labels, pred_labels,
                    target_names=self.data_module.splits['known_classes'],
                    digits=4
                ))

            # Save CSV version
            report_df.to_csv(save_path.with_suffix('.csv'))

        return report_df

    def reliability_diagram(self, logits, true_labels, n_bins=12, save_path=None):
        """Compute Expected Calibration Error and plot reliability diagram"""
        probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracy = (predictions == np.array(true_labels)).astype(np.float32)

        # Bin the confidence scores
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(confidence, bin_boundaries) - 1

        ece = 0.0
        bin_accuracies = []
        bin_confidences = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.any():
                bin_accuracy = accuracy[mask].mean()
                bin_confidence = confidence[mask].mean()
                bin_weight = mask.mean()

                ece += bin_weight * abs(bin_accuracy - bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)

        # Plot reliability diagram
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Model calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram\nECE = {ece:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return ece

    def compute_ood_metrics(self, val_loader, ood_loader):
        """Compute Out-of-Distribution detection metrics"""
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
            'energy_in': energy_in,
            'energy_out': energy_out,
            'fpr': fpr,
            'tpr': tpr
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

    def generate_tsne_plot(self, data_loader, ood_loader=None, save_path=None):
        """Generate t-SNE visualization of features"""
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
            n_iter=1000
        )

        embeddings = tsne.fit_transform(features[:self.config.TSNE_MAX_ITEMS])
        labels_tsne = labels[:self.config.TSNE_MAX_ITEMS]

        # Plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

        for i, class_name in enumerate(class_names):
            mask = labels_tsne == i
            if mask.any():
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                            c=[colors[i]], label=class_name, alpha=0.7, s=20)

        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Feature Embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return embeddings

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
