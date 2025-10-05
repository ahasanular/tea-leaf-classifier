import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as tvutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from einops import rearrange


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
        """Generates and saves prototype overlay visualizations."""
        path = self.config.EXPORT_DIR / "fig1_prototype_overlays.png"
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
