import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from einops import rearrange

from default_config import Config

import torch.nn as nn
from prototype_head import PrototypeHead


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction"""

    def __init__(self, name='efficientnet-b0', pretrained=True):
        super().__init__()
        if pretrained:
            self.net = EfficientNet.from_pretrained(name)
        else:
            self.net = EfficientNet.from_name(name)

        # Freeze early layers for faster training
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze early layers to prevent overfitting"""
        for param in list(self.net.parameters())[:100]:
            param.requires_grad = False

    def forward(self, x):
        return self.net.extract_features(x)

    def get_output_channels(self, input_size=(3, Config.IMG_SIZE, Config.IMG_SIZE)):
        """Get number of output channels for the backbone"""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            if Config.DEVICE != 'cpu':
                dummy = dummy.to(Config.DEVICE)
                self.net = self.net.to(Config.DEVICE)
            features = self.forward(dummy)
            return features.shape[1]

class TeaLeafModel(nn.Module):
    """Complete tea-leaf classification model with backbone and prototype head"""

    def __init__(self, num_classes, config):
        super().__init__()

        self.config = config
        self.num_classes = num_classes

        if self.config.DEVICE == 'mps':
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

        # Backbone
        self.backbone = EfficientNetBackbone(
            name=config.BACKBONE_NAME,
            pretrained=True
        )

        # Get output channels for the head
        in_channels = self.backbone.get_output_channels()

        # Prototype-based head
        self.head = PrototypeHead(
            in_channels=in_channels,
            num_classes=num_classes,
            protos_per_class=config.PROTOS_PER_CLASS,
            proto_dim=config.PROTOTYPE_DIM
        )

    def forward(self, x):
        features = self.backbone(x)
        logits, sim_max, additional = self.head(features)
        return logits, sim_max, additional

    def push_prototypes(self, data_loader):
        """Push prototypes using current model state"""
        self.head.push_prototypes(
            self.backbone,
            data_loader,
            self.config.DEVICE
        )
