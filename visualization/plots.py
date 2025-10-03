import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


class VisualizationEngine:
    """Handles all model visualization and explanation generation"""

    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = config.DEVICE

    def denormalize_image(self, tensor):
        """Convert normalized tensor back to image"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)[:, None, None]
        return (tensor * std + mean).clamp(0, 1)

    def plot_training_curves(self, history, save_path=None):
        """Plot training and validation accuracy curves"""
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history['lr'], label='Learning Rate', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_prototype_overlays(self, dataset, num_samples=6, save_path=None):
        """Create prototype activation overlays for model explanations"""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        overlay_rows = []
        samples_collected = 0

        with torch.no_grad():
            for x, y, path in dataloader:
                if samples_collected >= num_samples:
                    break

                x = x.to(self.device)
                logits, _, (similarity, spatial_dims, features) = self.model(x)
                prediction = logits.argmax(1).item()

                # Get prototypes for predicted class
                start_idx = prediction * self.config.PROTOS_PER_CLASS
                end_idx = (prediction + 1) * self.config.PROTOS_PER_CLASS

                # Reduce features
                reduced_features = self.model.head.reduce(features)
                B, D, H, W = reduced_features.shape
                flattened_features = reduced_features.view(B, D, -1).transpose(1, 2)
                flattened_features_norm = F.normalize(flattened_features, dim=-1)

                # Get prototype similarities
                prototypes_norm = F.normalize(self.model.head.prototype_vectors[start_idx:end_idx], dim=-1)
                spatial_similarity = torch.einsum('bnd,pd->bnp', flattened_features_norm, prototypes_norm)[0]

                # Get top-k prototypes
                prototype_scores = spatial_similarity.max(dim=0).values
                topk_indices = torch.topk(prototype_scores,
                                          k=min(self.config.TOPK_PROTOTYPE_OVERLAYS,
                                                self.config.PROTOS_PER_CLASS)).indices.cpu().tolist()

                # Create overlay row
                original_image = self.denormalize_image(x[0]).cpu()
                row_images = [original_image]

                for proto_idx in topk_indices:
                    # Create heatmap
                    heatmap = spatial_similarity[:, proto_idx].reshape(H, W).cpu().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                    # Resize heatmap to match image size
                    heatmap_resized = F.interpolate(
                        torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
                        size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0].numpy()

                    # Create overlay
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(original_image.permute(1, 2, 0).numpy())
                    ax.imshow(heatmap_resized, cmap='jet', alpha=0.6)
                    ax.axis('off')
                    ax.set_title(f'Proto {proto_idx}', fontsize=8)

                    # Convert plot to tensor
                    fig.canvas.draw()
                    plot_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                    plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    plot_array = plot_array[:, :, 1:4]  # ARGB to RGB
                    plt.close(fig)

                    plot_tensor = torch.tensor(plot_array).permute(2, 0, 1).float() / 255.0
                    row_images.append(plot_tensor)

                # Combine row images
                row_tensor = torch.cat([
                    F.interpolate(img.unsqueeze(0),
                                  size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
                                  mode='bilinear',
                                  align_corners=False).squeeze(0)
                    for img in row_images
                ], dim=2)

                overlay_rows.append(row_tensor)
                samples_collected += 1

        # Create final grid
        if overlay_rows:
            grid = tvutils.make_grid(overlay_rows, nrow=1, padding=10)
            if save_path:
                tvutils.save_image(grid, save_path)
                print(f"[Visualization] Saved prototype overlays to {save_path}")

            return grid
        return None

    def export_prototype_tiles(self, dataset, save_dir=None):
        """Export image patches that activate each prototype"""
        self.model.eval()

        if save_dir is None:
            save_dir = self.config.EXPORT_DIR / "prototype_tiles"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Track best matches for each prototype
        prototype_matches = [[] for _ in range(self.model.head.num_protos)]

        with torch.no_grad():
            for x, y, path in dataloader:
                x = x.to(self.device)
                features = self.model.backbone(x)
                reduced_features = self.model.head.reduce(features)

                B, D, H, W = reduced_features.shape
                flattened_features = reduced_features.view(B, D, -1).transpose(1, 2)
                flattened_features_norm = F.normalize(flattened_features, dim=-1)
                prototypes_norm = F.normalize(self.model.head.prototype_vectors, dim=-1)

                similarity = torch.einsum('bnd,pd->bnp', flattened_features_norm, prototypes_norm)[0]

                for proto_idx in range(self.model.head.num_protos):
                    max_sim_idx = similarity[:, proto_idx].argmax().item()
                    max_similarity = similarity[max_sim_idx, proto_idx].item()

                    h_pos = max_sim_idx // W
                    w_pos = max_sim_idx % W

                    prototype_matches[proto_idx].append(
                        (max_similarity, path[0], h_pos, w_pos)
                    )

        # Create tiles for each class
        cell_height = self.config.IMG_SIZE // H
        cell_width = self.config.IMG_SIZE // W

        for class_idx, class_name in enumerate(self.data_module.splits['known_classes']):
            start_idx = class_idx * self.config.PROTOS_PER_CLASS
            end_idx = (class_idx + 1) * self.config.PROTOS_PER_CLASS

            all_tiles = []

            for proto_idx in range(start_idx, end_idx):
                # Get top matches for this prototype
                top_matches = sorted(prototype_matches[proto_idx],
                                     key=lambda x: x[0], reverse=True)[:self.config.N_TILES_PER_PROTO]

                for similarity_score, image_path, h, w in top_matches:
                    # Load and process image
                    image = Image.open(image_path).convert("RGB")
                    image_resized = image.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))

                    # Calculate crop coordinates
                    center_x = int((w + 0.5) * cell_width)
                    center_y = int((h + 0.5) * cell_height)
                    half_size = self.config.TILE_PX // 2

                    # Ensure crop is within image bounds
                    x0 = max(0, center_x - half_size)
                    y0 = max(0, center_y - half_size)
                    x1 = min(self.config.IMG_SIZE, center_x + half_size)
                    y1 = min(self.config.IMG_SIZE, center_y + half_size)

                    # Adjust for non-square crops
                    crop_width = x1 - x0
                    crop_height = y1 - y0

                    if crop_width != crop_height:
                        size = min(crop_width, crop_height)
                        # Re-center the crop
                        center_x_adj = x0 + crop_width // 2
                        center_y_adj = y0 + crop_height // 2
                        x0 = max(0, center_x_adj - size // 2)
                        y0 = max(0, center_y_adj - size // 2)
                        x1 = min(self.config.IMG_SIZE, x0 + size)
                        y1 = min(self.config.IMG_SIZE, y0 + size)

                    # Crop and resize to consistent size
                    crop = image_resized.crop((x0, y0, x1, y1))
                    crop_resized = crop.resize((self.config.TILE_PX, self.config.TILE_PX),
                                               Image.Resampling.LANCZOS)

                    tile_tensor = torch.tensor(np.array(crop_resized)).permute(2, 0, 1).float() / 255.0
                    all_tiles.append(tile_tensor)

            # Create grid for this class
            if all_tiles:
                grid = tvutils.make_grid(all_tiles, nrow=self.config.N_TILES_PER_PROTO, padding=2)
                filename = save_dir / f"class_{class_name}_prototypes.png"
                tvutils.save_image(grid, filename)
                print(f"[Visualization] Saved prototype tiles for {class_name} to {filename}")

    def plot_ood_analysis(self, ood_metrics, save_path=None):
        """Plot OOD detection results"""
        if ood_metrics is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Curve
        ax1.plot(ood_metrics['fpr'], ood_metrics['tpr'], linewidth=2,
                 label=f'ROC (AUC = {ood_metrics["auroc"]:.3f})')
        ax1.plot([0, 1], [0, 1], '--', color='gray', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('OOD Detection ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy distributions
        ax2.hist(ood_metrics['energy_in'], bins=50, alpha=0.7,
                 label='In-distribution', density=True)
        ax2.hist(ood_metrics['energy_out'], bins=50, alpha=0.7,
                 label='Out-of-distribution', density=True)
        ax2.set_xlabel('Energy Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Energy Score Distributions\n(Higher = More OOD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
