import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PrototypeHead(nn.Module):
    """Interpretable prototype-based classification head"""

    def __init__(self, in_channels, num_classes, protos_per_class=12, proto_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.protos_per_class = protos_per_class
        self.num_protos = num_classes * protos_per_class

        # Feature reduction
        self.reduce = nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False)

        # Learnable prototypes
        self.prototype_vectors = nn.Parameter(torch.randn(self.num_protos, proto_dim))

        # Final classifier
        self.classifier = nn.Linear(self.num_protos, num_classes, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        # Initialize prototype vectors with Xavier uniform
        nn.init.xavier_uniform_(self.prototype_vectors)

        # Initialize classifier with prototype-friendly weights
        with torch.no_grad():
            self.classifier.weight.zero_()
            for c in range(self.num_classes):
                start_idx = c * self.protos_per_class
                end_idx = (c + 1) * self.protos_per_class
                self.classifier.weight[c, start_idx:end_idx] = 1.0 / self.protos_per_class

    def forward(self, features):
        """Forward pass with prototype similarity computation"""
        # Reduce feature dimensions
        z = self.reduce(features)
        B, D, H, W = z.shape

        # Flatten spatial dimensions
        z_flat = rearrange(z, 'b d h w -> b (h w) d')
        z_normalized = F.normalize(z_flat, dim=-1, p=2)

        # Normalize prototypes
        p_normalized = F.normalize(self.prototype_vectors, dim=-1, p=2)

        # Compute similarity between features and prototypes
        similarity = torch.einsum('bnd,pd->bnp', z_normalized, p_normalized)

        # Max similarity per prototype
        similarity_max, _ = similarity.max(dim=1)

        # Classification logits
        logits = self.classifier(similarity_max)

        return logits, similarity_max, (similarity, (H, W), z)

    @torch.no_grad()
    def push_prototypes(self, backbone, data_loader, device):
        """Update prototypes with the most similar feature patches"""
        self.eval()
        backbone.eval()

        P, D = self.prototype_vectors.shape
        best_similarity = torch.full((P,), -1e9, device=device)
        best_vectors = torch.zeros((P, D), device=device)

        for batch in data_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)

            # Extract features
            with torch.no_grad():
                features = backbone(x)
                z = self.reduce(features)
                B, D, H, W = z.shape
                z_flat = rearrange(z, 'b d h w -> b (h w) d')
                z_normalized = F.normalize(z_flat, dim=-1)
                p_normalized = F.normalize(self.prototype_vectors, dim=-1)
                similarity = torch.einsum('bnd,pd->bnp', z_normalized, p_normalized)

            # Update best matches
            for b in range(B):
                true_class = y[b].item()
                start_idx = true_class * self.protos_per_class
                end_idx = (true_class + 1) * self.protos_per_class

                # Get max similarity for this batch element's true class prototypes
                class_similarity = similarity[b, :, start_idx:end_idx].amax(dim=0)

                for j in range(self.protos_per_class):
                    proto_idx = start_idx + j
                    if class_similarity[j] > best_similarity[proto_idx]:
                        # Find the spatial location with highest similarity
                        hw_idx = similarity[b, :, proto_idx].argmax().item()
                        best_similarity[proto_idx] = class_similarity[j]
                        best_vectors[proto_idx] = z_flat[b, hw_idx, :].detach()

        # Update prototype vectors
        for p in range(P):
            if best_similarity[p] > -1e8:  # If we found a match
                self.prototype_vectors.data[p] = F.normalize(best_vectors[p], dim=0)
