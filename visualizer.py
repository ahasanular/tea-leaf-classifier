import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as tvutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from einops import rearrange
from pathlib import Path
import json
import pandas as pd
import os


class PrototypeOverlayVisualizer:
    """
    Visualizes the top prototype activations (heatmaps) over input images.
    Saves a single figure showing prototype overlays for several samples.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE

    @staticmethod
    def denorm_img(t):
        """Denormalize an image tensor from ImageNet normalization to [0,1]."""
        m = torch.tensor([0.485, 0.456, 0.406], device=t.device)[:, None, None]
        s = torch.tensor([0.229, 0.224, 0.225], device=t.device)[:, None, None]
        return (t * s + m).clamp(0, 1)

    def visualize_prototypes(self, dataset, samples=6):
        os.makedirs(self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / "plots", exist_ok=True)
        """Generates and saves prototype overlay visualizations."""
        path = self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / "plots/fig1_prototype_overlays.png"
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        rows = []
        count = 0

        head = self.model.head
        backbone = self.model.backbone

        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                feat = backbone(x)
                logits, sim_max, (sim_full, HW, z) = head(feat)
                pred = logits.argmax(1).item()
                start, end = pred * self.config.PROTOS_PER_CLASS, (pred + 1) * self.config.PROTOS_PER_CLASS

                # compute spatial similarities
                z_red = head.reduce(feat)
                zf = rearrange(z_red, 'b d h w -> b (h w) d')
                zf_n = F.normalize(zf, dim=-1)
                p_n = F.normalize(head.prototype_vectors[start:end], dim=-1)
                sim_spatial = torch.einsum('bnd,pd->bnp', zf_n, p_n)[0]
                H, W = HW

                sims = sim_max[0, start:end]
                topk = torch.topk(sims, k=min(self.config.TOPK_PROTOTYPE_OVERLAYS, self.config.PROTOS_PER_CLASS)).indices.cpu().tolist()

                base = self.denorm_img(x[0]).cpu()
                row = [tvutils.make_grid(base, nrow=1)]

                for j in topk:
                    heat = sim_spatial[:, j].reshape(H, W).cpu().numpy()
                    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                    heat_t = torch.tensor(heat)[None, None, :, :]
                    heat_up = F.interpolate(
                        heat_t, size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
                        mode="bilinear", align_corners=False
                    )[0, 0].numpy()

                    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
                    canvas = FigureCanvas(fig)
                    ax.imshow(base.permute(1, 2, 0).numpy())
                    ax.imshow(heat_up, cmap='jet', alpha=0.45)
                    ax.axis('off')
                    fig.tight_layout(pad=0)
                    canvas.draw()

                    # --- robust buffer extraction ---
                    if hasattr(canvas, "buffer_rgba"):
                        buf = np.asarray(canvas.buffer_rgba())
                    else:
                        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        w, h = canvas.get_width_height()
                        buf = buf.reshape(h, w, 3)

                    plt.close(fig)

                    # ✅ drop alpha if present
                    if buf.shape[-1] == 4:
                        buf = buf[..., :3]

                    row.append(torch.tensor(buf).permute(2, 0, 1).float() / 255.0)

                # concatenate visualizations horizontally
                row_cat = torch.cat([
                    F.interpolate(r.unsqueeze(0), size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
                                  mode='bilinear', align_corners=False).squeeze(0)
                    for r in row
                ], dim=2)

                rows.append(row_cat)
                count += 1
                if count >= samples:
                    break

        grid = tvutils.make_grid(rows, nrow=1)
        tvutils.save_image(grid, path)
        print(f"[Explainability] ✅ Saved prototype overlays → {path}")


class OODRocVisualizer:
    """Visualizes OOD detection results."""
    def __init__(self, config):
        self.config = config

    def generate(self, result):
        os.makedirs(self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / "plots", exist_ok=True)
        fpr = result['fpr']
        tpr = result['tpr']

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], '--', color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"OOD ROC (unknown: {self.config.UNKNOWN_CLASS_NAME})\nAUROC={result['auroc']:.3f}, FPR@95TPR={result['fpr_95']:.3f}")
        plt.tight_layout()
        plt.savefig(self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / f"plots/fig_roc_ood.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hist(result['energy_in'], bins=40, alpha=0.6, label="ID (val)")
        plt.hist(result['energy_out'], bins=40, alpha=0.6, label=f"OOD ({self.config.UNKNOWN_CLASS_NAME})")
        plt.xlabel("Energy score  (higher ⇒ more OOD)")
        plt.ylabel("Count")
        plt.title("Energy Score Distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.EXPORT_DIR / self.config.UNKNOWN_CLASS_NAME / f"plots/fig_energy_hist.png", dpi=300)
        plt.close()




class OODComparisonTable:
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(self.config.EXPORT_DIR)
        self.results = []

    def _load_json(self, file_path: Path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read {file_path}: {e}")
            return None

    def collect_results(self):
        # Traverse each subfolder
        for subdir in self.base_dir.iterdir():
            if not subdir.is_dir():
                continue

            json_path = subdir / "training_results.json"
            if not json_path.exists():
                continue

            data = self._load_json(json_path)
            if not data:
                continue

            # Extract relevant info
            test = data.get("test_results", {}).get("overall_metrics", {})
            unknown = data.get("class_information", {}).get("unknown_class", "N/A")
            known_classes = data.get("class_information", {}).get("known_classes", [])
            cfg = data.get("training_config", {}).get("model", {})

            self.results.append({
                "OOD_Class": unknown.replace("4. ", "").strip(),
                "Known_Classes": len(known_classes),
                "Accuracy": round(test.get("accuracy", 0), 4),
                "Precision": round(test.get("precision", 0), 4),
                "Recall": round(test.get("recall", 0), 4),
                "F1_Score": round(test.get("f1_score", 0), 4),
                "ECE": round(test.get("ece", 0), 4),
                "Backbone": cfg.get("backbone", "N/A"),
                "Prototypes_per_Class": cfg.get("protos_per_class", "N/A"),
            })

    def to_dataframe(self):
        if not self.results:
            self.collect_results()
        return pd.DataFrame(self.results)

    def save_csv(self):
        df = self.to_dataframe()
        df.to_csv(self.config.EXPORT_DIR / "ood-comparison.csv", index=False)
        print(f"✅ Saved comparison table to {self.config.EXPORT_DIR}")

    def show_summary(self):
        df = self.to_dataframe()
        print("\n📊 OOD Experiment Comparison:\n")
        print(df.to_string(index=False))
        print("\nAverage F1:", round(df["F1_Score"].mean(), 4))
        print("Average Accuracy:", round(df["Accuracy"].mean(), 4))


