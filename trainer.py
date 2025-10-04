import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import math
import time
from collections import defaultdict


class CosineWarmupScheduler:
    """Learning rate scheduler with warmup and cosine decay"""

    def __init__(self, optimizer, base_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        """Update learning rate for current epoch"""
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1
        return lr

    def _get_lr(self):
        """Compute learning rate for current epoch"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * float(self.current_epoch + 1) / float(max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = self.current_epoch - self.warmup_epochs
            total_decay = max(1, self.total_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress / total_decay))


class PrototypeTrainer:
    """Main trainer class for prototype-based classification"""

    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = config.DEVICE

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        self.criterion = None

        # History tracking
        self.history = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_state = None

        self.results_logger = None

        self._setup_training()

    def _setup_training(self):
        """Setup optimizer, scheduler, and criterion"""
        # Get class weights for imbalanced data
        class_weights = self.data_module.get_class_weights().to(self.device)

        # Combined loss: CrossEntropy + Prototype diversity regularization
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with all model parameters
        params = list(self.model.backbone.parameters()) + list(self.model.head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            self.config.LEARNING_RATE,
            self.config.WARMUP_EPOCHS,
            self.config.EPOCHS
        )

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y, _) in enumerate(self.data_module.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast():
                logits, sim_max, _ = self.model(x)
                # Combined loss: classification + prototype diversity
                cls_loss = self.criterion(logits, y)
                div_loss = 1e-4 * (sim_max ** 2).mean()  # Encourage diverse prototypes
                loss = cls_loss + div_loss

            # Backward pass with gradient clipping
            self.scaler.scale(loss).backward()
            if self.config.GRAD_CLIP:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(self.data_module.train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.data_module.train_loader)

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        all_logits = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for x, y, paths in self.data_module.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _ = self.model(x)

                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
                all_paths.extend(paths)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, all_logits, all_labels, all_paths

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        start_time = time.time()

        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()

            # Update learning rate
            current_lr = self.scheduler.step()

            # Train one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Push prototypes periodically
            if (epoch + 1) % self.config.PUSH_EVERY_EPOCH == 0:
                print("  > Pushing prototypes...")
                self.model.head.push_prototypes(
                    self.model.backbone,
                    self.data_module.train_loader,
                    self.device
                )

            # Validate
            val_acc, val_logits, val_labels, val_paths = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {
                    'backbone': self.model.backbone.state_dict(),
                    'head': self.model.head.state_dict(),
                    'known_classes': self.data_module.splits['known_classes'],
                    'config': self.config.__dict__,
                    'epoch': epoch,
                    'val_acc': val_acc
                }

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch + 1:03d}/{self.config.EPOCHS} | '
                  f'Time: {epoch_time:.1f}s | '
                  f'LR: {current_lr:.2e} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.3f} | '
                  f'Val Acc: {val_acc:.3f}')

            if self.results_logger:
                self.results_logger.log_training_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    lr=current_lr,
                    prototype_push=(epoch + 1) % self.config.PUSH_EVERY_EPOCH == 0
                )

            # Early stopping check (optional)
            if epoch > 10 and val_acc < 0.1:  # If model is not learning
                print("[WARNING] Model shows poor performance. Stopping early.")
                break



        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.3f}")

        return self.history, self.best_state
