import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
import gc

try:
    from neptunecontrib.monitoring.sacred import NeptuneObserver

    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

# Sacred experiment setup
ex = Experiment('sadaan')
ex.observers.append(MongoObserver(url='localhost:27017', db_name='sadaan'))

# Import your previous implementations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sadaan import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


@ex.config
def config():
    """Sacred configuration"""
    dataset_path = './synthetic_medical_dataset'
    config_path = None

    model = {
        'in_channels': 1,             # CT/MRI grayscale input
        'num_structures': 5,          # number of anatomical classes
        'presence_threshold': 0.5     # threshold for absence detection
        # ⚠️ no feature_channels here — encoder decides output channels
    }

    training = {
        'batch_size': 4,
        'learning_rate': 5e-4,
        'num_epochs': 100,
        'early_stopping_patience': 20,
        'gradient_clip_norm': 0.5,
        'num_workers': 2,
        'warmup_epochs': 5,
        'validate_every_n_epochs': 2,
        'gradient_accumulation_steps': 2,
        'memory_cleanup_frequency': 5,
        'max_history_length': 50
    }

    loss_weights = {
        'segmentation': 1.0,
        'dice': 2.0,
        'focal_seg': 1.0,
        'absence': 1.0,
        'attention_supervision': 0.5,
        'confidence': 0.1
    }

    augmentation = {
        'noise_std': 2.0,
        'intensity_scale_range': [0.95, 1.05],
        'enabled': True,
        'rotation_degrees': 5,
        'flip_probability': 0.3
    }

    optimizer = {
        'type': 'adamw',
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine_annealing',
        'lr_min': 1e-6,
        'lr_restart_period': 20,
        'lr_factor': 0.5,
        'lr_patience': 10
    }

    checkpoint = {
        'save_dir': './checkpoints',
        'save_frequency': 10,
        'keep_best_only': False
    }

    logging = {
        'log_frequency': 1,
        'visualize_predictions': True,
        'num_visualization_samples': 3
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42


class DatasetLoader:
    """Load pre-generated synthetic datasets from disk"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_path = self.dataset_path / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def load_split(self, split: str) -> Dict:
        """Load a specific data split (train/val/test)"""
        split_path = self.dataset_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split {split} not found at {split_path}")

        data_path = split_path / 'data.npz'
        data = np.load(data_path)

        scenarios_path = split_path / 'scenarios.json'
        with open(scenarios_path, 'r') as f:
            scenarios = json.load(f)

        return {
            'images': data['images'],
            'masks': data['masks'],
            'presence_labels': data['presence_labels'],
            'scenarios': scenarios
        }

    def get_structure_names(self) -> List[str]:
        """Get list of structure names"""
        return self.metadata['structure_names']

    def get_image_size(self) -> Tuple[int, int, int]:
        """Get image dimensions"""
        return tuple(self.metadata['image_size'])

    def print_dataset_info(self):
        """Print dataset information"""
        print("Dataset Information:")
        print(f"  Path: {self.dataset_path}")
        print(f"  Structures: {len(self.get_structure_names())}")
        print(f"  Image size: {self.get_image_size()}")
        print(f"  Generation params: {self.metadata.get('generation_params', 'N/A')}")

        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if split_path.exists():
                data = self.load_split(split)
                print(f"  {split.capitalize()}: {len(data['images'])} samples")

                from collections import Counter
                scenario_counts = Counter(data['scenarios'])
                print(
                    f"    Scenarios: {dict(list(scenario_counts.items())[:5])}{'...' if len(scenario_counts) > 5 else ''}")



class AugmentationTransform:
    """Simple augmentation transforms for medical images"""

    def __init__(self, noise_std=5.0, intensity_scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, image):
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        if self.intensity_scale_range != (1.0, 1.0):
            scale_min, scale_max = self.intensity_scale_range
            scale = torch.empty(1).uniform_(scale_min, scale_max).item()
            image = image * scale

        image = torch.clamp(image, 0, 255)
        return image


class MetricsCalculator:
    """Calculate and track various metrics for evaluation"""

    def __init__(self, structure_names: List[str]):
        self.structure_names = structure_names
        self.num_structures = len(structure_names)

    def dice_coefficient(self, pred_mask, true_mask, epsilon=1e-6):
        """Calculate Dice coefficient"""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()

        intersection = (pred_flat * true_flat).sum()
        return (2.0 * intersection + epsilon) / (pred_flat.sum() + true_flat.sum() + epsilon)

    def calculate_segmentation_metrics(self, pred_masks, true_masks, presence_labels):
        """Calculate segmentation metrics only for present structures"""
        metrics = {}
        batch_size = pred_masks.shape[0]

        pred_masks_binary = (pred_masks > 0.5).float()

        structure_dice = {}
        for i, struct_name in enumerate(self.structure_names):
            dice_scores = []
            for b in range(batch_size):
                if presence_labels[b, i] == 1:
                    dice = self.dice_coefficient(
                        pred_masks_binary[b, i],
                        true_masks[b, i]
                    )
                    dice_scores.append(dice.item())

            if dice_scores:
                structure_dice[struct_name] = {
                    'mean': np.mean(dice_scores),
                    'std': np.std(dice_scores),
                    'count': len(dice_scores)
                }

        metrics['structure_dice'] = structure_dice

        all_dice = [score['mean'] for score in structure_dice.values()]
        if all_dice:
            metrics['mean_dice'] = np.mean(all_dice)
            metrics['std_dice'] = np.std(all_dice)
        else:
            metrics['mean_dice'] = 0.0
            metrics['std_dice'] = 0.0

        return metrics

    def calculate_presence_metrics(self, pred_presence, true_presence):
        """Calculate presence detection metrics"""
        pred_presence_binary = (pred_presence > 0.5).long()

        structure_accuracy = {}
        for i, struct_name in enumerate(self.structure_names):
            correct = (pred_presence_binary[:, i] == true_presence[:, i]).float()
            structure_accuracy[struct_name] = correct.mean().item()

        overall_accuracy = (pred_presence_binary == true_presence).float().mean().item()

        present_mask = (true_presence == 1)
        absent_mask = (true_presence == 0)

        if present_mask.any():
            present_accuracy = (pred_presence_binary[present_mask] == 1).float().mean().item()
        else:
            present_accuracy = 0.0

        if absent_mask.any():
            absent_accuracy = (pred_presence_binary[absent_mask] == 0).float().mean().item()
        else:
            absent_accuracy = 0.0

        return {
            'structure_accuracy': structure_accuracy,
            'overall_accuracy': overall_accuracy,
            'present_accuracy': present_accuracy,
            'absent_accuracy': absent_accuracy,
            'mean_structure_accuracy': np.mean(list(structure_accuracy.values()))
        }


class LearningRateWarmup:
    """Learning rate warmup scheduler"""

    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


@ex.capture
def create_model(model, structure_names, image_size, device):
    """Create and initialize model"""
    model = SpatialAttentionMedicalSegmenter(
        in_channels=model['in_channels'],
        num_structures=len(structure_names),
        feature_channels=model['feature_channels'],
        spatial_dims=image_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ex.log_scalar('model.total_parameters', total_params)
    ex.log_scalar('model.trainable_parameters', trainable_params)

    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return model


@ex.capture
def create_data_loaders(dataset_path, training, augmentation):
    """Load datasets and create data loaders"""
    loader = DatasetLoader(dataset_path)
    loader.print_dataset_info()

    structure_names = loader.get_structure_names()
    image_size = loader.get_image_size()

    train_data = loader.load_split('train')
    val_data = loader.load_split('val')

    train_transform = None
    if augmentation['enabled']:
        train_transform = AugmentationTransform(
            noise_std=augmentation['noise_std'],
            intensity_scale_range=tuple(augmentation['intensity_scale_range'])
        )

    train_dataset = SyntheticMedicalDataset(train_data, transform=train_transform)
    val_dataset = SyntheticMedicalDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training['batch_size'],
        shuffle=True,
        num_workers=training['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if training['num_workers'] > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training['batch_size'],
        shuffle=False,
        num_workers=training['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if training['num_workers'] > 0 else False,
        drop_last=False
    )

    ex.log_scalar('data.train_samples', len(train_dataset))
    ex.log_scalar('data.val_samples', len(val_dataset))
    ex.log_scalar('data.num_structures', len(structure_names))
    ex.info['data.structure_names'] = structure_names
    ex.info['data.image_size'] = image_size

    return train_loader, val_loader, structure_names, image_size


def cleanup_memory():
    """Enhanced memory cleanup"""
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            torch.cuda.memory._record_memory_history()
        except:
            pass


class Trainer:
    """Main training class with comprehensive memory optimization"""

    @ex.capture
    def __init__(self, model, train_loader, val_loader, structure_names,
                 training, loss_weights, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.structure_names = structure_names
        self.device = device
        self.training_config = training

        self.loss_fn = SpatialAttentionLoss(
            structure_names,
            weights=loss_weights,
            focal_alpha=0.25,
            focal_gamma=2.0
        )

        self.optimizer = self.create_optimizer(model, training, optimizer)

        if optimizer['lr_scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=optimizer['lr_factor'],
                patience=optimizer['lr_patience']
            )
        elif optimizer['lr_scheduler'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=optimizer.get('lr_restart_period', 20),
                eta_min=optimizer.get('lr_min', 1e-6)
            )
        else:
            self.scheduler = None

        self.metrics_calc = MetricsCalculator(structure_names)

        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 1)
        self.memory_cleanup_frequency = training.get('memory_cleanup_frequency', 5)
        self.max_history_length = training.get('max_history_length', 50)

        # FIX: Initialize minimal history to prevent unbounded growth
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_presence_acc': [],
            'val_presence_acc': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }

        self.global_step = 0
        warmup_epochs = training.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            self.setup_warmup_scheduler(warmup_epochs)

        self.current_epoch = 0

        if torch.cuda.is_available():
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def dice_coefficient_fast(self, pred_mask, true_mask, epsilon=1e-6):
        """Fast dice calculation without creating intermediate tensors"""
        pred_flat = pred_mask.view(-1)
        true_flat = true_mask.view(-1)
        intersection = (pred_flat * true_flat).sum()
        dice = (2.0 * intersection + epsilon) / (pred_flat.sum() + true_flat.sum() + epsilon)
        return dice

    def handle_oom(self):
        """Handle out of memory errors by increasing gradient accumulation"""
        if hasattr(self, 'gradient_accumulation_steps'):
            self.gradient_accumulation_steps *= 2
            print(f"OOM detected, increasing gradient accumulation to {self.gradient_accumulation_steps}")
            cleanup_memory()
            return True
        return False

    @ex.capture
    def train_epoch(self, logging) -> Dict:
        """Train for one epoch with comprehensive memory optimization"""
        self.model.train()

        # FIX: Use running totals instead of accumulating lists
        running_losses = defaultdict(float)
        loss_counts = 0
        running_presence_correct = 0
        running_presence_total = 0
        running_dice_sum = 0
        running_dice_count = 0

        # Memory tracking
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        warmup_epochs = getattr(self, 'warmup_epochs', 0)

        if hasattr(self, 'warmup_scheduler') and hasattr(self, 'current_epoch'):
            if self.current_epoch < warmup_epochs:
                self.warmup_scheduler.step()
                print(
                    f"Warmup epoch {self.current_epoch + 1}/{warmup_epochs}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # FIX: Remove tqdm progress bar to prevent tensor retention
        total_batches = len(self.train_loader)
        accumulated_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Print progress without storing objects
            if batch_idx % max(1, total_batches // 10) == 0:
                print(f"Training progress: {batch_idx}/{total_batches}")

            self.global_step += 1

            try:
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                true_masks = batch['masks'].to(self.device, non_blocking=True)
                true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

                # Forward pass with autocast for memory efficiency
                if self.gradient_accumulation_steps > 1 and batch_idx % self.gradient_accumulation_steps != 0:
                    pass
                else:
                    self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)

                targets = {
                    'segmentation_targets': true_masks,
                    'presence_targets': true_presence
                }

                losses = self.loss_fn(outputs, targets)
                total_loss = losses['total']

                if self.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.gradient_accumulation_steps

                # Backward pass
                total_loss.backward()
                accumulated_loss += total_loss.item()

                # Update weights if gradient accumulation step is complete
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.training_config['gradient_clip_norm']
                    )
                    self.optimizer.step()

                    if hasattr(self, 'global_step') and self.global_step % (logging.get('log_frequency', 1) * 10) == 0:
                        ex.log_scalar('train.gradient_norm', grad_norm.item(), self.global_step)

                    accumulated_loss = 0

                # FIX: Accumulate losses as running totals instead of lists
                for key, value in losses.items():
                    loss_value = value.item() if hasattr(value, 'item') else float(value)
                    running_losses[key] += loss_value
                loss_counts += 1

                # Sacred logging
                if self.global_step % logging.get('log_frequency', 1) == 0:
                    for key, value in losses.items():
                        loss_value = value.item() if hasattr(value, 'item') else float(value)
                        ex.log_scalar(f'train.batch.{key}', loss_value, self.global_step)

                    current_lr = self.optimizer.param_groups[0]['lr']
                    ex.log_scalar('train.learning_rate', current_lr, self.global_step)

                    if torch.cuda.is_available():
                        memory_gb = torch.cuda.memory_allocated() / 1024 ** 3
                        ex.log_scalar('system.gpu_memory_gb', memory_gb, self.global_step)

                # Debug prints for first few batches
                if batch_idx < 3:
                    print(f"Batch {batch_idx} Debug:")
                    for key, value in losses.items():
                        loss_value = value.item() if hasattr(value, 'item') else float(value)
                        print(f"  {key}: {loss_value:.4f}")

                    if batch_idx == 0:
                        with torch.no_grad():
                            seg_probs = outputs['segmentation_probs']
                            seg_mean = seg_probs.mean().item()
                            seg_max = seg_probs.max().item()
                            seg_min = seg_probs.min().item()

                            near_zero = (seg_probs < 0.01).float().mean().item()
                            near_one = (seg_probs > 0.99).float().mean().item()

                            print(f"  Seg Probs: mean={seg_mean:.4f}, min={seg_min:.4f}, max={seg_max:.4f}")
                            print(f"  Saturation: {near_zero * 100:.1f}% near 0, {near_one * 100:.1f}% near 1")

                            presence_probs = outputs['presence_probs']
                            true_presence_count = true_presence.sum().item()
                            pred_presence_count = (presence_probs > 0.5).sum().item()

                            print(f"  Presence: True={true_presence_count}, Pred={pred_presence_count}")

                # FIX: Calculate metrics incrementally
                with torch.no_grad():
                    pred_presence_binary = (outputs['presence_probs'] > 0.5).long()
                    presence_correct = (pred_presence_binary == true_presence).float().sum()
                    running_presence_correct += presence_correct.item()
                    running_presence_total += true_presence.numel()

                    for b in range(true_presence.shape[0]):
                        for s in range(true_presence.shape[1]):
                            if true_presence[b, s] == 1:
                                pred_mask = outputs['segmentation_probs'][b, s]
                                true_mask = true_masks[b, s]
                                dice = self.dice_coefficient_fast(pred_mask, true_mask)
                                running_dice_sum += dice.item()
                                running_dice_count += 1

                # FIX: Complete tensor cleanup with explicit deletion
                del outputs, losses, total_loss, targets
                del images, true_masks, true_presence, batch

                # Force cleanup every few batches
                if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                    cleanup_memory()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch {batch_idx}, attempting recovery...")
                    if self.handle_oom():
                        continue
                    else:
                        raise e
                else:
                    raise e

        # FIX: Calculate final metrics from running totals
        if loss_counts > 0 and running_presence_total > 0:
            avg_losses = {k: v / loss_counts for k, v in running_losses.items()}
            presence_accuracy = running_presence_correct / running_presence_total
            mean_dice = running_dice_sum / running_dice_count if running_dice_count > 0 else 0.0

            presence_metrics = {'overall_accuracy': presence_accuracy, 'mean_structure_accuracy': presence_accuracy}
            seg_metrics = {'mean_dice': mean_dice, 'std_dice': 0.0}
        else:
            avg_losses = {'total': 0.0}
            presence_metrics = {'overall_accuracy': 0.0, 'mean_structure_accuracy': 0.0}
            seg_metrics = {'mean_dice': 0.0, 'std_dice': 0.0}

        cleanup_memory()

        train_results = {
            'losses': avg_losses,
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        print(f"\nEpoch Training Summary:")
        print(f"  Total Loss: {train_results['losses']['total']:.4f}")
        print(f"  Segmentation Loss: {train_results['losses'].get('segmentation', 0):.4f}")
        print(f"  Dice Loss: {train_results['losses'].get('dice', 0):.4f}")
        print(f"  Presence Accuracy: {presence_metrics['overall_accuracy']:.4f}")
        print(f"  Mean Dice: {seg_metrics['mean_dice']:.4f}")

        return train_results

    def setup_warmup_scheduler(self, warmup_epochs=5):
        """Setup learning rate warmup scheduler"""
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = LearningRateWarmup(
            self.optimizer,
            warmup_epochs,
            self.training_config['learning_rate']
        )
        print(f"Warmup scheduler initialized for {warmup_epochs} epochs")

    def create_optimizer(self, model, training, optimizer_config):
        """Create optimizer with support for AdamW"""
        if optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=training['learning_rate']
            )
        elif optimizer_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=training['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=training['learning_rate'],
                momentum=0.9,
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

        return optimizer

    def validate_epoch(self) -> Dict:
        """Validate for one epoch with comprehensive memory optimization"""
        self.model.eval()

        # FIX: Use running totals instead of lists
        running_losses = defaultdict(float)
        loss_counts = 0
        running_val_presence_correct = 0
        running_val_presence_total = 0
        running_val_dice_sum = 0
        running_val_dice_count = 0

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            total_batches = len(self.val_loader)

            for batch_idx, batch in enumerate(self.val_loader):
                # Print progress without storing progress objects
                if batch_idx % max(1, total_batches // 10) == 0:
                    print(f"Validation progress: {batch_idx}/{total_batches}")

                try:
                    # Move to device
                    images = batch['image'].to(self.device, non_blocking=True)
                    true_masks = batch['masks'].to(self.device, non_blocking=True)
                    true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

                    # Forward pass with autocast
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = self.model(images)

                    # Calculate loss
                    targets = {
                        'segmentation_targets': true_masks,
                        'presence_targets': true_presence
                    }
                    losses = self.loss_fn(outputs, targets)

                    # FIX: Accumulate losses as running totals
                    for key, value in losses.items():
                        loss_value = value.item() if hasattr(value, 'item') else float(value)
                        running_losses[key] += loss_value
                    loss_counts += 1

                    # Calculate metrics incrementally
                    pred_presence_binary = (outputs['presence_probs'] > 0.5).long()
                    presence_correct = (pred_presence_binary == true_presence).float().sum()
                    running_val_presence_correct += presence_correct.item()
                    running_val_presence_total += true_presence.numel()

                    # Dice score for present structures only
                    batch_size, num_structures = true_presence.shape
                    for b in range(batch_size):
                        for s in range(num_structures):
                            if true_presence[b, s] == 1:
                                pred_mask = outputs['segmentation_probs'][b, s]
                                true_mask = true_masks[b, s]
                                dice = self.dice_coefficient_fast(pred_mask, true_mask)
                                running_val_dice_sum += dice.item()
                                running_val_dice_count += 1

                    # FIX: Complete tensor cleanup with explicit deletion
                    del outputs, losses, targets, pred_presence_binary, presence_correct
                    del images, true_masks, true_presence, batch

                    # More frequent memory cleanup during validation
                    if (batch_idx + 1) % max(1, self.memory_cleanup_frequency // 2) == 0:
                        cleanup_memory()

                        # Monitor memory growth
                        if torch.cuda.is_available():
                            current_memory = torch.cuda.memory_allocated()
                            memory_growth = (current_memory - initial_memory) / 1024 ** 3
                            if memory_growth > 1.0:
                                print(f"Warning: Validation memory growth: {memory_growth:.2f} GB")
                                for _ in range(5):
                                    gc.collect()
                                torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM during validation at batch {batch_idx}")
                        cleanup_memory()
                        continue
                    else:
                        raise e

        # Calculate final metrics from running totals
        if loss_counts > 0 and running_val_presence_total > 0:
            val_presence_accuracy = running_val_presence_correct / running_val_presence_total
            val_mean_dice = running_val_dice_sum / running_val_dice_count if running_val_dice_count > 0 else 0.0

            avg_losses = {k: v / loss_counts for k, v in running_losses.items()}

            presence_metrics = {
                'overall_accuracy': val_presence_accuracy,
                'mean_structure_accuracy': val_presence_accuracy
            }
            seg_metrics = {
                'mean_dice': val_mean_dice,
                'std_dice': 0.0
            }
        else:
            avg_losses = {k: 0.0 for k in running_losses.keys()}
            presence_metrics = {'overall_accuracy': 0.0, 'mean_structure_accuracy': 0.0}
            seg_metrics = {'mean_dice': 0.0, 'std_dice': 0.0}

        # Final cleanup
        del running_losses
        cleanup_memory()

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            total_growth = (final_memory - initial_memory) / 1024 ** 3
            if total_growth > 0.1:
                print(f"Validation memory growth: {total_growth:.2f} GB")

        val_results = {
            'losses': avg_losses,
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        return val_results

    def limit_history_size(self):
        """Limit history to prevent unbounded memory growth"""
        max_len = self.max_history_length
        if len(self.history['train_loss']) > max_len:
            for key in self.history:
                if isinstance(self.history[key], list):
                    self.history[key] = self.history[key][-max_len:]

    @ex.capture
    def train(self, training, checkpoint):
        """Main training loop with comprehensive memory optimization"""
        num_epochs = training['num_epochs']
        early_stopping_patience = training['early_stopping_patience']
        validate_every_n_epochs = training.get('validate_every_n_epochs', 1)

        save_path = Path(checkpoint['save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training_config['batch_size']}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.training_config['batch_size'] * self.gradient_accumulation_steps}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            if torch.cuda.is_available():
                print(f"GPU memory before epoch: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            # Train
            train_results = self.train_epoch()

            # FIX: Only validate every N epochs to save memory
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_results = self.validate_epoch()
            else:
                # Use previous validation results or defaults
                val_results = {
                    'losses': {'total': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')},
                    'presence_metrics': {'overall_accuracy': self.history['val_presence_acc'][-1] if self.history[
                        'val_presence_acc'] else 0.0},
                    'segmentation_metrics': {
                        'mean_dice': self.history['val_dice'][-1] if self.history['val_dice'] else 0.0}
                }

            val_loss = val_results['losses']['total']
            if self.scheduler:
                # Handle different scheduler types correctly
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
                else:
                    self.scheduler.step()  # CosineAnnealingLR and others don't take arguments

            current_lr = self.optimizer.param_groups[0]['lr']

            # FIX: Store only essential metrics as scalars, not complex objects
            self.history['train_loss'].append(train_results['losses']['total'])
            self.history['val_loss'].append(val_loss)
            self.history['train_presence_acc'].append(train_results['presence_metrics']['overall_accuracy'])
            self.history['val_presence_acc'].append(val_results['presence_metrics']['overall_accuracy'])
            self.history['train_dice'].append(train_results['segmentation_metrics']['mean_dice'])
            self.history['val_dice'].append(val_results['segmentation_metrics']['mean_dice'])
            self.history['learning_rates'].append(current_lr)

            # FIX: Limit history size to prevent unbounded growth
            self.limit_history_size()

            # Sacred logging
            ex.log_scalar('train.epoch.total_loss', train_results['losses']['total'], epoch)
            ex.log_scalar('val.epoch.total_loss', val_loss, epoch)
            ex.log_scalar('train.epoch.presence_accuracy', train_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('val.epoch.presence_accuracy', val_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('train.epoch.mean_dice', train_results['segmentation_metrics']['mean_dice'], epoch)
            ex.log_scalar('val.epoch.mean_dice', val_results['segmentation_metrics']['mean_dice'], epoch)
            ex.log_scalar('epoch.learning_rate', current_lr, epoch)

            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024 ** 3
                ex.log_scalar('system.epoch_end_memory_gb', memory_gb, epoch)

            # Log detailed loss components
            for loss_name, loss_value in train_results['losses'].items():
                if loss_name != 'total':
                    ex.log_scalar(f'train.epoch.{loss_name}_loss', loss_value, epoch)

            for loss_name, loss_value in val_results['losses'].items():
                if loss_name != 'total':
                    ex.log_scalar(f'val.epoch.{loss_name}_loss', loss_value, epoch)

            # Print epoch summary
            print(f"Train Loss: {train_results['losses']['total']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train Presence Acc: {train_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Val Presence Acc: {val_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Train Mean Dice: {train_results['segmentation_metrics']['mean_dice']:.4f}")
            print(f"Val Mean Dice: {val_results['segmentation_metrics']['mean_dice']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            if torch.cuda.is_available():
                print(f"GPU memory after epoch: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            # Save checkpoint
            if (epoch + 1) % checkpoint['save_frequency'] == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch)

            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_model_path = save_path / 'best_model.pth'
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                print(f"New best model saved! Val loss: {val_loss:.4f}")

                ex.log_scalar('best.val_loss', val_loss, epoch)
                ex.log_scalar('best.epoch', epoch, epoch)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                ex.log_scalar('training.stopped_early', 1)
                ex.log_scalar('training.final_epoch', epoch)
                break

            # FIX: Clear results objects to prevent accumulation
            del train_results, val_results
            cleanup_memory()

        ex.log_scalar('training.completed_epochs', epoch + 1)
        ex.log_scalar('training.best_val_loss', best_val_loss)

        print("\nTraining completed!")
        cleanup_memory()

        return self.history

    def save_checkpoint(self, filepath: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint with aggressive memory optimization"""
        # FIX: Create minimal checkpoint and clean up immediately
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'structure_names': self.structure_names,
            'is_best': is_best,
            # FIX: Only save essential scalars, not full history
            'best_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf'),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else float('inf')
        }

        torch.save(checkpoint_data, filepath)
        ex.add_artifact(str(filepath), f'checkpoint_epoch_{epoch + 1}.pth')

        # FIX: Immediate cleanup of checkpoint data
        del checkpoint_data
        cleanup_memory()

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint with memory optimization"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']

        # FIX: Clean up loaded checkpoint immediately
        del checkpoint
        cleanup_memory()

        return epoch


def visualize_predictions(model, dataset, structure_names, device='cuda', num_samples=3):
    """Visualize model predictions with comprehensive memory optimization"""
    model.eval()

    # FIX: Limit samples to prevent memory issues
    num_samples = min(num_samples, 2)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # FIX: Process samples one at a time instead of creating large figure
    for sample_idx, idx in enumerate(indices):
        # Create individual figure for each sample
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        with torch.no_grad():
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            true_masks = sample['masks'].numpy()
            true_presence = sample['presence_labels'].numpy()
            scenario = sample['scenario']

            # Get model predictions with autocast
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(image)

            pred_presence = outputs['presence_probs'][0].cpu().numpy()
            pred_masks = outputs['segmentation_probs'][0].cpu().numpy()
            attention_maps = outputs['attention_maps'][0].cpu().numpy()

            # FIX: Clean up GPU tensors immediately after copying to CPU
            del outputs

            slice_idx = image.shape[-1] // 2
            img_slice = image[0, 0, :, :, slice_idx].cpu().numpy()

            # FIX: Delete image tensor immediately
            del image

            # Plot visualizations
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title(f'Original\n{scenario}')
            axes[0].axis('off')

            # True segmentation overlay
            true_overlay = img_slice.copy()
            for j in range(len(structure_names)):
                if true_presence[j] == 1:
                    mask_slice = true_masks[j, :, :, slice_idx]
                    true_overlay[mask_slice > 0.5] = 255

            axes[1].imshow(true_overlay, cmap='gray')
            axes[1].set_title('True Segmentation')
            axes[1].axis('off')

            # Predicted segmentation overlay
            pred_overlay = img_slice.copy()
            for j in range(len(structure_names)):
                if pred_presence[j] > 0.5:
                    mask_slice = pred_masks[j, :, :, slice_idx]
                    pred_overlay[mask_slice > 0.5] = 255

            axes[2].imshow(pred_overlay, cmap='gray')
            axes[2].set_title('Pred Segmentation')
            axes[2].axis('off')

            # Attention map
            avg_attention = np.mean(attention_maps, axis=0)
            attention_slice = avg_attention[:, :, slice_idx]
            attention_slice = np.rot90(attention_slice)
            axes[3].imshow(img_slice, cmap='gray', alpha=0.7)
            axes[3].imshow(attention_slice, cmap='hot', alpha=0.5)
            axes[3].set_title('Attention Map')
            axes[3].axis('off')

            # Print presence predictions
            print(f"\nSample {sample_idx + 1} ({scenario}):")
            print("Structure | True | Pred | Conf")
            print("-" * 35)
            for j, struct_name in enumerate(structure_names):
                true_val = "Present" if true_presence[j] == 1 else "Absent"
                pred_val = "Present" if pred_presence[j] > 0.5 else "Absent"
                conf = pred_presence[j]
                print(f"{struct_name:12} | {true_val:7} | {pred_val:7} | {conf:.3f}")

        plt.tight_layout()

        # FIX: Save each figure individually and clean up immediately
        fig_path = f'./temp_visualization_{sample_idx}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        ex.add_artifact(fig_path, f'predictions_visualization_{sample_idx}.png')
        plt.show()

        # FIX: Immediate cleanup after each sample
        plt.close(fig)
        del fig, axes, img_slice, true_overlay, pred_overlay, avg_attention, attention_slice
        del true_masks, pred_masks, attention_maps, pred_presence, true_presence
        cleanup_memory()


@ex.capture
def plot_training_history(history: Dict, logging):
    """Plot training curves with memory optimization"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # FIX: Use simplified history structure
    axes[0, 1].plot(epochs, history['train_presence_acc'], 'b-', label='Train Presence Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_presence_acc'], 'r-', label='Val Presence Acc', linewidth=2)
    axes[0, 1].set_title('Presence Detection Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[1, 0].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[1, 0].set_title('Mean Dice Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    curves_path = './temp_training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    ex.add_artifact(curves_path, 'training_curves.png')

    if logging['visualize_predictions']:
        plt.show()

    # FIX: Clean up plot objects
    plt.close(fig)
    del fig, axes
    cleanup_memory()


def analyze_structure_performance(history, structure_names):
    """Analyze performance with memory optimization"""
    if len(history['train_loss']) == 0:
        print("No metrics available for analysis")
        return {}

    # FIX: Use only final values instead of full history analysis
    final_train_pres = history['train_presence_acc'][-1] if history['train_presence_acc'] else 0.0
    final_val_pres = history['val_presence_acc'][-1] if history['val_presence_acc'] else 0.0
    final_train_dice = history['train_dice'][-1] if history['train_dice'] else 0.0
    final_val_dice = history['val_dice'][-1] if history['val_dice'] else 0.0

    print("\nFinal Performance Summary:")
    print("=" * 50)
    print(f"Final Train Presence Acc: {final_train_pres:.4f}")
    print(f"Final Val Presence Acc: {final_val_pres:.4f}")
    print(f"Final Train Dice: {final_train_dice:.4f}")
    print(f"Final Val Dice: {final_val_dice:.4f}")

    structure_performance = {
        'overall': {
            'train_presence_acc': final_train_pres,
            'val_presence_acc': final_val_pres,
            'train_dice': final_train_dice,
            'val_dice': final_val_dice
        }
    }

    ex.log_scalar('final.overall.presence_acc', final_val_pres)
    ex.log_scalar('final.overall.dice', final_val_dice)
    ex.info['final_structure_performance'] = structure_performance

    return structure_performance


@ex.capture
def save_results(history, structure_performance, checkpoint):
    """Save training results with memory optimization"""
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)

    # FIX: Save minimal essential data only
    essential_data = {
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
        'final_train_dice': history['train_dice'][-1] if history['train_dice'] else 0.0,
        'final_val_dice': history['val_dice'][-1] if history['val_dice'] else 0.0,
        'final_train_presence': history['train_presence_acc'][-1] if history['train_presence_acc'] else 0.0,
        'final_val_presence': history['val_presence_acc'][-1] if history['val_presence_acc'] else 0.0,
        'num_epochs': len(history['train_loss'])
    }

    # Save only recent loss curves (last 20 epochs) to reduce file size
    recent_history = {
        'train_loss': history['train_loss'][-20:],
        'val_loss': history['val_loss'][-20:],
        'learning_rates': history['learning_rates'][-20:]
    }

    history_path = results_path / 'essential_training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(recent_history, f)

    perf_path = results_path / 'structure_performance.json'
    with open(perf_path, 'w') as f:
        json.dump(structure_performance, f, indent=2)

    # Save summary
    summary_path = results_path / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(essential_data, f, indent=2)

    ex.add_artifact(str(history_path), 'essential_training_history.pkl')
    ex.add_artifact(str(perf_path), 'structure_performance.json')
    ex.add_artifact(str(summary_path), 'training_summary.json')

    print(f"Results saved to {results_path}")

    # FIX: Clean up data objects
    del essential_data, recent_history
    cleanup_memory()


@ex.automain
def main(dataset_path, device, seed, logging, config_path):
    """Main training pipeline with comprehensive memory optimization"""

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        print(f"Loaded additional config from: {config_path}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ex.log_scalar('system.device', str(device))

    # FIX: Enhanced CUDA optimization
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass

        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    print("Loading datasets...")
    train_loader, val_loader, structure_names, image_size = create_data_loaders()
    print(f"Loaded dataset with {len(structure_names)} structures")
    print(f"Structure names: {structure_names}")

    print("Creating model...")
    model = create_model(structure_names=structure_names, image_size=image_size, device=device)

    if torch.cuda.is_available():
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # FIX: Create trainer with memory monitoring
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        structure_names=structure_names,
        device=device
    )

    print("Starting training...")
    history = trainer.train()

    print("Analyzing results...")
    structure_performance = analyze_structure_performance(history, structure_names)

    # FIX: Clean up trainer to free model memory before plotting
    del trainer, model
    cleanup_memory()

    plot_training_history(history)

    # FIX: Only visualize if explicitly requested and with minimal samples
    if logging['visualize_predictions']:
        print("Visualizing predictions...")
        # Recreate minimal model for visualization only
        vis_model = create_model(structure_names=structure_names, image_size=image_size, device=device)
        # Load best weights if available
        best_model_path = Path('./checkpoints/best_model.pth')
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            vis_model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint

        val_dataset = val_loader.dataset
        visualize_predictions(
            vis_model, val_dataset, structure_names, device=device,
            num_samples=1  # FIX: Minimal samples only
        )

        # FIX: Clean up visualization model
        del vis_model
        cleanup_memory()

    save_results(history, structure_performance)

    # FIX: Calculate final metrics from simplified history
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    final_val_acc = history['val_presence_acc'][-1] if history['val_presence_acc'] else 0.0
    final_val_dice = history['val_dice'][-1] if history['val_dice'] else 0.0

    ex.log_scalar('final.validation_loss', final_val_loss)
    ex.log_scalar('final.presence_accuracy', final_val_acc)
    ex.log_scalar('final.mean_dice', final_val_dice)

    # FIX: Final comprehensive cleanup
    del history, structure_performance, train_loader, val_loader
    cleanup_memory()

    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    print("Training completed successfully!")

    return {
        'final_val_loss': final_val_loss,
        'final_presence_accuracy': final_val_acc,
        'final_mean_dice': final_val_dice
    }


if __name__ == "__main__":
    import sys

    print("Sacred Training Script - Comprehensive Memory Optimization")
    print("Usage examples:")
    print("python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/'")
    print(
        "python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/' config_path='../config/config.json'")
    print("\nMemory optimization features:")
    print("- Running totals instead of tensor accumulation")
    print("- Immediate tensor cleanup after use")
    print("- Progress tracking without object retention")
    print("- Simplified history storage (scalars only)")
    print("- Per-sample visualization processing")
    print("- Aggressive garbage collection")
    print("- Limited validation frequency")
    print("- Minimal checkpoint data")
    print("- Enhanced CUDA memory management")
    print("\nStarting experiment...")

    if len(sys.argv) == 1:
        ex.run()
    else:
        pass