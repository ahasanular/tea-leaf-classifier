import re
import numpy as np
from pathlib import Path
from collections import Counter
from difflib import get_close_matches

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets
from PIL import Image

from default_config import Config
from preprocessing import get_transforms


class NameResolver:
    """Handles class name resolution and aliases"""

    _ALIAS_BANK = {
        'helopeltis': ['tea mosquito bug', 'mosquito bug', 'helopeltis antonii', 'tea_mosquito_bug',
                       'tea-mosquito-bug'],
        'brownblight': ['brown blight', 'blight (brown)', 'brown_blight'],
        'grayblight': ['grey blight', 'gray blight', 'grey_blight', 'gray_blight'],
        'redspider': ['red spider', 'red spider mite', 'redspider mite', 'mite'],
        'greenmiridbug': ['green mirid bug', 'mirid bug', 'mirid'],
        'algalspot': ['algal leaf spot', 'tea algal leaf spot', 'algalspot'],
        'healthy': ['fresh', 'normal']
    }

    @staticmethod
    def _normalize_name(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    @classmethod
    def resolve_unknown_name(cls, requested, available):
        """Resolve class names with aliases and fuzzy matching"""
        if not requested:
            return None

        for c in available:
            if c.lower() == requested.lower():
                return c

        reqn = cls._normalize_name(requested)
        norm_map = {cls._normalize_name(c): c for c in available}

        if reqn in norm_map:
            return norm_map[reqn]

        for c in available:
            if reqn in cls._normalize_name(c) or cls._normalize_name(c) in reqn:
                return c

        for key, vals in cls._ALIAS_BANK.items():
            if reqn == key or any(reqn == cls._normalize_name(v) for v in vals):
                for c in available:
                    if any(cls._normalize_name(v) in cls._normalize_name(c) for v in [requested] + vals):
                        return c

        cand = get_close_matches(requested, available, n=1, cutoff=0.6)
        return cand[0] if cand else None


class TeaLeafDataSplitter:
    """Handles dataset splitting and OOD configuration"""

    def __init__(self, data_root, unknown_class=None, val_ratio=0.15, seed=Config.SEED):
        self.data_root = Path(data_root)
        self.unknown_class = unknown_class
        self.val_ratio = val_ratio
        self.seed = seed

        if not self.data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    def discover_classes(self):
        """Discover available classes in the dataset"""
        dataset = datasets.ImageFolder(self.data_root)
        print(f"[INFO] Discovered {len(dataset.classes)} classes: {dataset.classes}")
        return dataset.classes, dataset

    def build_splits(self):
        """Build train/val splits with OOD handling"""
        all_classes, full_dataset = self.discover_classes()

        # Resolve unknown class name
        resolved_unknown = NameResolver.resolve_unknown_name(
            self.unknown_class, all_classes
        ) if self.unknown_class else None

        # Handle unknown class resolution
        if self.unknown_class and resolved_unknown is None:
            counts = Counter([t for _, t in full_dataset.samples])
            rarest_id = min(counts, key=lambda k: counts[k])
            resolved_unknown = all_classes[rarest_id]
            print(f"[WARN] Requested unknown '{self.unknown_class}' not found.")
            print(f"       Available classes: {all_classes}")
            print(f"       Auto-selecting rarest class as unknown: '{resolved_unknown}'")

        # Separate known and unknown samples
        known_files, known_labels = [], []
        for path, target in full_dataset.samples:
            cls_name = all_classes[target]
            if resolved_unknown is not None and cls_name == resolved_unknown:
                continue
            known_files.append(path)
            known_labels.append(target)

        known_labels = np.array(known_labels)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
        train_idx, val_idx = next(sss.split(known_files, known_labels))

        train_files = [known_files[i] for i in train_idx]
        val_files = [known_files[i] for i in val_idx]
        train_labels = known_labels[train_idx]
        val_labels = known_labels[val_idx]

        # OOD files
        ood_files = []
        if resolved_unknown is not None:
            unk_id = all_classes.index(resolved_unknown)
            ood_files = [p for p, t in full_dataset.samples if t == unk_id]
            if len(ood_files) == 0:
                print(f"[WARN] Resolved unknown '{resolved_unknown}' has 0 images. OOD will be skipped.")
                resolved_unknown = None

        # Remap known class indices
        known_ids = sorted(set(known_labels.tolist()))
        id_remap = {old: i for i, old in enumerate(known_ids)}
        known_classes = [all_classes[i] for i in known_ids]

        print(f"[SPLIT] Train={len(train_files)}  Val={len(val_files)}  "
              f"OOD({resolved_unknown if resolved_unknown else 'None'})={len(ood_files)}")
        print(f"[SPLIT] Known classes: {known_classes}")

        return {
            "all_classes": all_classes,
            "known_classes": known_classes,
            "id_remap": id_remap,
            "train_files": train_files,
            "val_files": val_files,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "ood_files": ood_files,
            "resolved_unknown": resolved_unknown
        }


class TeaLeafDataset(Dataset):
    """Custom Dataset for tea leaf images"""

    def __init__(self, paths, labels=None, transform=None, id_remap=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.id_remap = id_remap

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.paths[idx]).convert("RGB")

        # Apply transformations
        img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)

        # Handle labels
        if self.labels is None:
            return img_t, -1, self.paths[idx]

        y_old = self.labels[idx]
        return img_t, self.id_remap[y_old], self.paths[idx]


class DataModule:
    """Main data module that orchestrates all data operations"""

    def __init__(self, config):
        self.config = config
        self.splitter = TeaLeafDataSplitter(
            config.DATA_ROOT,
            config.UNKNOWN_CLASS_NAME,
            config.VAL_RATIO,
            config.SEED
        )
        self.splits = None
        self.train_loader = None
        self.val_loader = None
        self.val_ds = None
        self.ood_loader = None

    def prepare_data(self):
        """Prepare all data splits and loaders"""
        self.splits = self.splitter.build_splits()

        # Get transforms
        train_tfms, test_tfms = get_transforms(self.config.IMG_SIZE)

        # Create datasets
        train_ds = TeaLeafDataset(
            self.splits['train_files'],
            self.splits['train_labels'],
            train_tfms,
            self.splits['id_remap']
        )

        self.val_ds = TeaLeafDataset(
            self.splits['val_files'],
            self.splits['val_labels'],
            test_tfms,
            self.splits['id_remap']
        )

        ood_ds = None
        if len(self.splits['ood_files']) > 0:
            ood_ds = TeaLeafDataset(
                self.splits['ood_files'],
                None,
                test_tfms,
                self.splits['id_remap']
            )

        # Create samplers for class imbalance
        train_sampler = self._get_balanced_sampler(
            self.splits['train_labels']) if self.config.USE_OVERSAMPLING else None

        # Create data loaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.ood_loader = DataLoader(
            ood_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ) if ood_ds else None

        return self.splits

    def _get_balanced_sampler(self, labels):
        """Create a balanced sampler to address class imbalance"""
        class_counts = Counter([self.splits['id_remap'][y] for y in labels])
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[self.splits['id_remap'][label]] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    def get_class_weights(self):
        """Calculate class weights for loss function"""
        counts = Counter([self.splits['id_remap'][y] for y in self.splits['train_labels']])
        K = len(set(self.splits['id_remap'].values()))
        total = sum(counts.values())

        freqs = np.array([counts.get(i, 0) for i in range(K)], dtype=np.float32) / max(1, total)
        inv = 1.0 / np.clip(freqs, 1e-6, None)
        weights = inv / inv.sum() * K

        if self.config.CLASS_WEIGHT_SMOOTHING > 0:
            weights = (1 - self.config.CLASS_WEIGHT_SMOOTHING) * weights + \
                      self.config.CLASS_WEIGHT_SMOOTHING * np.ones_like(weights)

        return torch.tensor(weights, dtype=torch.float32)
