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
# Add default observer - will be updated in main if sacred_dir is provided
# ex.observers.append(FileStorageObserver('./sacred_logs'))
ex.observers.append(MongoObserver(url='localhost:27017', db_name='sadaan'))

# Import your previous implementations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sadaan import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


@ex.config
def config():
    """Sacred configuration"""
    # Data parameters
    dataset_path = './synthetic_medical_dataset'
    # sacred_dir = './sacred_logs'
    config_path = None

    # Model parameters
    model = {
        'in_channels': 1,
        'feature_channels': 64,
        'presence_threshold': 0.5
    }
    # Training parameters
    training = {
        'batch_size': 4,
        'learning_rate': 5e-4,
        'num_epochs': 100,
        'early_stopping_patience': 20,
        'gradient_clip_norm': 0.5,
        'num_workers': 2,
        'warmup_epochs': 5,
        'validate_every_n_epochs': 2,
        # Memory optimization parameters
        'gradient_accumulation_steps': 2,  # For simulating larger batches
        'memory_cleanup_frequency': 10,  # Clean memory every N batches
        'max_batches_in_memory': 50  # Limit cached data
    }
    # Loss parameters
    loss_weights = {
        'segmentation': 1.0,
        'dice': 2.0,
        'focal_seg': 1.0,
        'absence': 1.0,
        'attention_supervision': 0.5,
        'confidence': 0.1
    }
    # Augmentation parameters
    augmentation = {
        'noise_std': 2.0,
        'intensity_scale_range': [0.95, 1.05],
        'enabled': True,
        'rotation_degrees': 5,
        'flip_probability': 0.3
    }
    # Optimizer parameters
    optimizer = {
        'type': 'adamw',
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine_annealing',
        'lr_min': 1e-6,
        'lr_restart_period': 20,
        'lr_factor': 0.5,
        'lr_patience': 10
    }
    # Checkpoint parameters
    checkpoint = {
        'save_dir': './checkpoints',
        'save_frequency': 10,
        'keep_best_only': False
    }
    # Logging parameters
    logging = {
        'log_frequency': 1,  # Log every N batches
        'visualize_predictions': True,
        'num_visualization_samples': 3
    }
    # Hardware
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

        # Load compressed numpy data
        data_path = split_path / 'data.npz'
        data = np.load(data_path)

        # Load scenarios
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

        # Print split sizes
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if split_path.exists():
                data = self.load_split(split)
                print(f"  {split.capitalize()}: {len(data['images'])} samples")

                # Scenario breakdown
                from collections import Counter
                scenario_counts = Counter(data['scenarios'])
                print(
                    f"    Scenarios: {dict(list(scenario_counts.items())[:5])}{'...' if len(scenario_counts) > 5 else ''}")


class SyntheticMedicalDataset(Dataset):
    """PyTorch Dataset wrapper for pre-loaded synthetic medical data"""

    def __init__(self, data_dict: Dict, transform=None):
        self.images = data_dict['images']
        self.masks = data_dict['masks']
        self.presence_labels = data_dict['presence_labels']
        self.scenarios = data_dict['scenarios']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        image = self.images[idx]  # [H, W, D]
        masks = self.masks[idx]  # [num_structures, H, W, D]
        presence = self.presence_labels[idx]  # [num_structures]
        scenario = self.scenarios[idx]

        # Convert to tensors
        image = torch.from_numpy(image).float()
        masks = torch.from_numpy(masks).float()
        presence = torch.from_numpy(presence).long()

        # Add channel dimension to image
        image = image.unsqueeze(0)  # [1, H, W, D]

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'masks': masks,
            'presence_labels': presence,
            'scenario': scenario,
            'index': idx
        }


class AugmentationTransform:
    """Simple augmentation transforms for medical images"""

    def __init__(self, noise_std=5.0, intensity_scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, image):
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Scale intensity
        if self.intensity_scale_range != (1.0, 1.0):
            scale_min, scale_max = self.intensity_scale_range
            scale = torch.empty(1).uniform_(scale_min, scale_max).item()
            image = image * scale

        # Clip to reasonable range
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

        # Per-structure metrics
        structure_dice = {}
        for i, struct_name in enumerate(self.structure_names):
            dice_scores = []
            for b in range(batch_size):
                if presence_labels[b, i] == 1:  # Only calculate for present structures
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

        # Overall metrics
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

        # Per-structure accuracy
        structure_accuracy = {}
        for i, struct_name in enumerate(self.structure_names):
            correct = (pred_presence_binary[:, i] == true_presence[:, i]).float()
            structure_accuracy[struct_name] = correct.mean().item()

        # Overall metrics
        overall_accuracy = (pred_presence_binary == true_presence).float().mean().item()

        # Per-class metrics (present vs absent)
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
            # Linear warmup
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

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ex.log_scalar('model.total_parameters', total_params)
    ex.log_scalar('model.trainable_parameters', trainable_params)

    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return model


@ex.capture
def create_data_loaders(dataset_path, training, augmentation):
    """Load datasets and create data loaders"""
    # Load dataset
    loader = DatasetLoader(dataset_path)
    loader.print_dataset_info()

    structure_names = loader.get_structure_names()
    image_size = loader.get_image_size()

    # Load splits
    train_data = loader.load_split('train')
    val_data = loader.load_split('val')

    # Create transforms
    train_transform = None
    if augmentation['enabled']:
        train_transform = AugmentationTransform(
            noise_std=augmentation['noise_std'],
            intensity_scale_range=tuple(augmentation['intensity_scale_range'])
        )

    # Create datasets
    train_dataset = SyntheticMedicalDataset(train_data, transform=train_transform)
    val_dataset = SyntheticMedicalDataset(val_data)

    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=training['batch_size'],
        shuffle=True,
        num_workers=training['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if training['num_workers'] > 0 else False,  # Keep workers alive
        drop_last=True  # Drop incomplete batches to ensure consistent memory usage
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

    # Log dataset info to Sacred
    ex.log_scalar('data.train_samples', len(train_dataset))
    ex.log_scalar('data.val_samples', len(val_dataset))
    ex.log_scalar('data.num_structures', len(structure_names))
    ex.info['data.structure_names'] = structure_names
    ex.info['data.image_size'] = image_size

    return train_loader, val_loader, structure_names, image_size


def cleanup_memory():
    """Force garbage collection and CUDA memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Trainer:
    """Main training class with Sacred logging integration and memory optimization"""

    @ex.capture
    def __init__(self, model, train_loader, val_loader, structure_names,
                 training, loss_weights, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.structure_names = structure_names
        self.device = device
        self.training_config = training

        # Use improved loss function instead of original
        self.loss_fn = SpatialAttentionLoss(
            structure_names,
            weights=loss_weights,
            focal_alpha=0.25,
            focal_gamma=2.0
        )

        # Fixed optimizer creation
        self.optimizer = self.create_optimizer(model, training, optimizer)

        # Learning rate scheduler
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

        # Metrics calculator
        self.metrics_calc = MetricsCalculator(structure_names)

        # Memory optimization parameters
        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 1)
        self.memory_cleanup_frequency = training.get('memory_cleanup_frequency', 10)

        # Training history - limit memory usage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # Sacred logging
        self.global_step = 0

        # Setup warmup if specified
        warmup_epochs = training.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            self.setup_warmup_scheduler(warmup_epochs)

        # Track current epoch for warmup
        self.current_epoch = 0

        # Initialize memory tracking
        if torch.cuda.is_available():
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    @ex.capture
    def train_epoch(self, logging) -> Dict:
        """Train for one epoch with improved loss handling and diagnostics"""
        self.model.train()
        epoch_losses = defaultdict(list)

        # Use lists with limited capacity for memory efficiency
        batch_pred_presence = []
        batch_true_presence = []
        batch_pred_masks = []
        batch_true_masks = []
        batch_true_presence_for_seg = []

        # Track memory usage
        max_batches_in_memory = self.training_config.get('max_batches_in_memory', 50)

        pbar = tqdm(self.train_loader, desc="Training")
        accumulated_loss = 0

        # Initialize warmup scheduler if in warmup phase
        warmup_epochs = getattr(self, 'warmup_epochs', 0)
        if hasattr(self, 'warmup_scheduler') and hasattr(self, 'current_epoch'):
            if self.current_epoch < warmup_epochs:
                self.warmup_scheduler.step()
                print(
                    f"Warmup epoch {self.current_epoch + 1}/{warmup_epochs}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        for batch_idx, batch in enumerate(pbar):
            self.global_step += 1

            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            true_masks = batch['masks'].to(self.device, non_blocking=True)
            true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

            # Forward pass
            if self.gradient_accumulation_steps > 1 and batch_idx % self.gradient_accumulation_steps != 0:
                # Don't zero gradients for gradient accumulation
                pass
            else:
                self.optimizer.zero_grad()

            outputs = self.model(images)

            # Prepare targets
            targets = {
                'segmentation_targets': true_masks,
                'presence_targets': true_presence
            }

            # Calculate improved loss with all components
            losses = self.loss_fn(outputs, targets)
            total_loss = losses['total']

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            total_loss.backward()
            accumulated_loss += total_loss.item()

            # Update weights if gradient accumulation step is complete
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.training_config['gradient_clip_norm']
                )

                self.optimizer.step()

                # Log gradient norm for monitoring
                if hasattr(self, 'global_step') and self.global_step % (logging.get('log_frequency', 1) * 10) == 0:
                    ex.log_scalar('train.gradient_norm', grad_norm.item(), self.global_step)

                accumulated_loss = 0

            # Store losses (only actual loss values, not scaled)
            for key, value in losses.items():
                # Handle both tensors and floats
                if hasattr(value, 'item'):
                    epoch_losses[key].append(value.item())
                else:
                    epoch_losses[key].append(float(value))

            # Enhanced Sacred logging with all loss components
            if self.global_step % logging.get('log_frequency', 1) == 0:
                for key, value in losses.items():
                    # Handle both tensors and floats for Sacred logging
                    loss_value = value.item() if hasattr(value, 'item') else float(value)
                    ex.log_scalar(f'train.batch.{key}', loss_value, self.global_step)

                # Log learning rate and memory usage
                current_lr = self.optimizer.param_groups[0]['lr']
                ex.log_scalar('train.learning_rate', current_lr, self.global_step)

                if torch.cuda.is_available():
                    memory_gb = torch.cuda.memory_allocated() / 1024 ** 3
                    ex.log_scalar('system.gpu_memory_gb', memory_gb, self.global_step)

            # Enhanced debug prints for first few batches with all loss components
            if batch_idx < 3:
                print(f"Batch {batch_idx} Debug:")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Segmentation Loss: {losses.get('segmentation', 0):.4f}")
                print(f"  Dice Loss: {losses.get('dice', 0):.4f}")
                print(f"  Focal Loss: {losses.get('focal_seg', 0):.4f}")
                print(f"  Presence Loss: {losses.get('absence', 0):.4f}")
                print(f"  Attention Loss: {losses.get('attention_supervision', 0):.4f}")
                print(f"  Confidence Loss: {losses.get('confidence', 0):.4f}")

                # Quick segmentation check for first batch
                if batch_idx == 0:
                    with torch.no_grad():
                        seg_probs = outputs['segmentation_probs']
                        seg_mean = seg_probs.mean().item()
                        seg_max = seg_probs.max().item()
                        seg_min = seg_probs.min().item()

                        # Check for saturation issues
                        near_zero = (seg_probs < 0.01).float().mean().item()
                        near_one = (seg_probs > 0.99).float().mean().item()

                        print(f"  Seg Probs: mean={seg_mean:.4f}, min={seg_min:.4f}, max={seg_max:.4f}")
                        print(f"  Saturation: {near_zero * 100:.1f}% near 0, {near_one * 100:.1f}% near 1")

                        # Quick presence check
                        presence_probs = outputs['presence_probs']
                        true_presence_count = true_presence.sum().item()
                        pred_presence_count = (presence_probs > 0.5).sum().item()

                        print(f"  Presence: True={true_presence_count}, Pred={pred_presence_count}")

            # Collect predictions for metrics (with memory management)
            with torch.no_grad():
                # Only keep predictions if we haven't exceeded memory limit
                if len(batch_pred_presence) < max_batches_in_memory:
                    batch_pred_presence.append(outputs['presence_probs'].detach().cpu())
                    batch_true_presence.append(true_presence.detach().cpu())
                    batch_pred_masks.append(outputs['segmentation_probs'].detach().cpu())
                    batch_true_masks.append(true_masks.detach().cpu())
                    batch_true_presence_for_seg.append(true_presence.detach().cpu())

            # Explicit cleanup of outputs and intermediate tensors
            del outputs, losses, total_loss, targets

            # Periodic memory cleanup
            if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                cleanup_memory()

            # Enhanced progress bar with more loss components
            avg_total_loss = np.mean(epoch_losses['total']) if epoch_losses['total'] else 0
            avg_dice_loss = np.mean(epoch_losses.get('dice', [0])) if epoch_losses.get('dice') else 0

            pbar.set_postfix({
                'total': f"{avg_total_loss:.4f}",
                'dice': f"{avg_dice_loss:.4f}",
                'mem_gb': f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}" if torch.cuda.is_available() else "N/A"
            })

            # Clear variables at end of batch
            del images, true_masks, true_presence

        # Calculate epoch metrics from collected predictions
        if batch_pred_presence:  # Only if we have collected predictions
            all_pred_presence = torch.cat(batch_pred_presence, dim=0)
            all_true_presence = torch.cat(batch_true_presence, dim=0)
            all_pred_masks = torch.cat(batch_pred_masks, dim=0)
            all_true_masks = torch.cat(batch_true_masks, dim=0)
            all_true_presence_for_seg = torch.cat(batch_true_presence_for_seg, dim=0)

            # Presence detection metrics
            presence_metrics = self.metrics_calc.calculate_presence_metrics(
                all_pred_presence, all_true_presence
            )

            # Segmentation metrics
            seg_metrics = self.metrics_calc.calculate_segmentation_metrics(
                all_pred_masks, all_true_masks, all_true_presence_for_seg
            )

            # Enhanced per-structure Dice reporting for debugging
            if hasattr(seg_metrics, 'structure_dice') and seg_metrics['structure_dice']:
                print("\nPer-structure Dice scores:")
                for struct_name, dice_info in seg_metrics['structure_dice'].items():
                    print(f"  {struct_name}: {dice_info['mean']:.4f} ± {dice_info['std']:.4f} (n={dice_info['count']})")

            # Clean up tensors
            del all_pred_presence, all_true_presence, all_pred_masks, all_true_masks, all_true_presence_for_seg
        else:
            # Fallback metrics if memory limit exceeded
            presence_metrics = {'overall_accuracy': 0.0, 'mean_structure_accuracy': 0.0}
            seg_metrics = {'mean_dice': 0.0}
            print("Warning: Metrics calculated on limited samples due to memory constraints")

        # Clear batch collections
        del batch_pred_presence, batch_true_presence, batch_pred_masks, batch_true_masks, batch_true_presence_for_seg

        # Final memory cleanup
        cleanup_memory()

        # Enhanced results with all loss components
        train_results = {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        # Print epoch summary with enhanced loss breakdown
        print(f"\nEpoch Training Summary:")
        print(f"  Total Loss: {train_results['losses']['total']:.4f}")
        print(f"  Segmentation Loss: {train_results['losses'].get('segmentation', 0):.4f}")
        print(f"  Dice Loss: {train_results['losses'].get('dice', 0):.4f}")
        print(f"  Focal Loss: {train_results['losses'].get('focal_seg', 0):.4f}")
        print(f"  Presence Accuracy: {presence_metrics['overall_accuracy']:.4f}")
        print(f"  Mean Dice: {seg_metrics['mean_dice']:.4f}")

        return train_results

    # Additional helper method to add to the Trainer class for warmup support
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
        """Validate for one epoch with memory optimization"""
        self.model.eval()
        epoch_losses = defaultdict(list)

        # Memory-efficient validation
        all_pred_presence = []
        all_true_presence = []
        all_pred_masks = []
        all_true_masks = []
        all_true_presence_for_seg = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                true_masks = batch['masks'].to(self.device, non_blocking=True)
                true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                targets = {
                    'segmentation_targets': true_masks,
                    'presence_targets': true_presence
                }
                losses = self.loss_fn(outputs, targets)

                # Store losses
                for key, value in losses.items():
                    epoch_losses[key].append(value.item())

                # Collect predictions (move to CPU immediately)
                all_pred_presence.append(outputs['presence_probs'].cpu())
                all_true_presence.append(true_presence.cpu())
                all_pred_masks.append(outputs['segmentation_probs'].cpu())
                all_true_masks.append(true_masks.cpu())
                all_true_presence_for_seg.append(true_presence.cpu())

                # Clean up GPU tensors immediately
                del outputs, losses, targets, images, true_masks, true_presence

                # Update progress bar
                avg_loss = np.mean(
                    [loss for batch_losses in epoch_losses.values() for loss in batch_losses]) if epoch_losses else 0
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'mem_gb': f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}" if torch.cuda.is_available() else "N/A"
                })

                # Periodic cleanup during validation
                if (batch_idx + 1) % (self.memory_cleanup_frequency * 2) == 0:
                    cleanup_memory()

        # Calculate metrics
        all_pred_presence = torch.cat(all_pred_presence, dim=0)
        all_true_presence = torch.cat(all_true_presence, dim=0)
        all_pred_masks = torch.cat(all_pred_masks, dim=0)
        all_true_masks = torch.cat(all_true_masks, dim=0)
        all_true_presence_for_seg = torch.cat(all_true_presence_for_seg, dim=0)

        presence_metrics = self.metrics_calc.calculate_presence_metrics(
            all_pred_presence, all_true_presence
        )

        seg_metrics = self.metrics_calc.calculate_segmentation_metrics(
            all_pred_masks, all_true_masks, all_true_presence_for_seg
        )

        # Clean up
        del all_pred_presence, all_true_presence, all_pred_masks, all_true_masks, all_true_presence_for_seg
        cleanup_memory()

        val_results = {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        return val_results

    @ex.capture
    def train(self, training, checkpoint):
        """Main training loop with memory optimization"""
        num_epochs = training['num_epochs']
        early_stopping_patience = training['early_stopping_patience']

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

            # Memory status before training
            if torch.cuda.is_available():
                print(f"GPU memory before epoch: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            # Train
            train_results = self.train_epoch()

            # Validate
            val_results = self.validate_epoch()

            # Update learning rate scheduler
            val_loss = val_results['losses']['total']
            if self.scheduler:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history (limit memory by not keeping too much detail)
            self.history['train_loss'].append(train_results['losses']['total'])
            self.history['val_loss'].append(val_loss)

            # Only keep essential metrics to save memory
            essential_train_metrics = {
                'presence_metrics': {'overall_accuracy': train_results['presence_metrics']['overall_accuracy']},
                'segmentation_metrics': {'mean_dice': train_results['segmentation_metrics']['mean_dice']}
            }
            essential_val_metrics = {
                'presence_metrics': {'overall_accuracy': val_results['presence_metrics']['overall_accuracy']},
                'segmentation_metrics': {'mean_dice': val_results['segmentation_metrics']['mean_dice']}
            }

            self.history['train_metrics'].append(essential_train_metrics)
            self.history['val_metrics'].append(essential_val_metrics)
            self.history['learning_rates'].append(current_lr)

            # Sacred logging (epoch-level)
            ex.log_scalar('train.epoch.total_loss', train_results['losses']['total'], epoch)
            ex.log_scalar('val.epoch.total_loss', val_loss, epoch)
            ex.log_scalar('train.epoch.presence_accuracy', train_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('val.epoch.presence_accuracy', val_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('train.epoch.mean_dice', train_results['segmentation_metrics']['mean_dice'], epoch)
            ex.log_scalar('val.epoch.mean_dice', val_results['segmentation_metrics']['mean_dice'], epoch)
            ex.log_scalar('epoch.learning_rate', current_lr, epoch)

            # Log memory usage
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

                # Save best model
                best_model_path = save_path / 'best_model.pth'
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                print(f"New best model saved! Val loss: {val_loss:.4f}")

                # Sacred logging
                ex.log_scalar('best.val_loss', val_loss, epoch)
                ex.log_scalar('best.epoch', epoch, epoch)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                ex.log_scalar('training.stopped_early', 1)
                ex.log_scalar('training.final_epoch', epoch)
                break

            # Memory cleanup at end of epoch
            cleanup_memory()

        # Final Sacred logging
        ex.log_scalar('training.completed_epochs', epoch + 1)
        ex.log_scalar('training.best_val_loss', best_val_loss)

        print("\nTraining completed!")

        # Final memory cleanup
        cleanup_memory()

        return self.history

    def save_checkpoint(self, filepath: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint with memory optimization"""
        # Create minimal checkpoint to save memory
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'structure_names': self.structure_names,
            'is_best': is_best,
            # Only save essential history to reduce checkpoint size
            'train_loss_history': self.history['train_loss'],
            'val_loss_history': self.history['val_loss']
        }

        torch.save(checkpoint, filepath)

        # Log checkpoint path to Sacred
        ex.add_artifact(str(filepath), f'checkpoint_epoch_{epoch + 1}.pth')

        # Clean up checkpoint from memory
        del checkpoint
        cleanup_memory()

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore essential history
        if 'train_loss_history' in checkpoint:
            self.history['train_loss'] = checkpoint['train_loss_history']
            self.history['val_loss'] = checkpoint['val_loss_history']

        epoch = checkpoint['epoch']

        # Clean up loaded checkpoint
        del checkpoint
        cleanup_memory()

        return epoch


def visualize_predictions(model, dataset, structure_names, device='cuda', num_samples=3):
    """Visualize model predictions on sample data with memory optimization"""
    model.eval()

    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]

            # Prepare input
            image = sample['image'].unsqueeze(0).to(device)  # Add batch dim
            true_masks = sample['masks'].numpy()
            true_presence = sample['presence_labels'].numpy()
            scenario = sample['scenario']

            # Get model predictions
            outputs = model(image)
            pred_presence = outputs['presence_probs'][0].cpu().numpy()  # Remove batch dim
            pred_masks = outputs['segmentation_probs'][0].cpu().numpy()
            attention_maps = outputs['attention_maps'][0].cpu().numpy()

            # Clean up GPU tensors immediately
            del outputs

            # Get middle slice for visualization
            slice_idx = image.shape[-1] // 2
            img_slice = image[0, 0, :, :, slice_idx].cpu().numpy()

            # Clean up image tensor
            del image

            # Plot original image
            axes[i, 0].imshow(img_slice, cmap='gray')
            axes[i, 0].set_title(f'Original\n{scenario}')
            axes[i, 0].axis('off')

            # Plot true segmentation overlay
            true_overlay = img_slice.copy()
            for j in range(len(structure_names)):
                if true_presence[j] == 1:
                    mask_slice = true_masks[j, :, :, slice_idx]
                    true_overlay[mask_slice > 0.5] = 255

            axes[i, 1].imshow(true_overlay, cmap='gray')
            axes[i, 1].set_title('True Segmentation')
            axes[i, 1].axis('off')

            # Plot predicted segmentation overlay
            pred_overlay = img_slice.copy()
            for j in range(len(structure_names)):
                if pred_presence[j] > 0.5:
                    mask_slice = pred_masks[j, :, :, slice_idx]
                    pred_overlay[mask_slice > 0.5] = 255

            axes[i, 2].imshow(pred_overlay, cmap='gray')
            axes[i, 2].set_title('Pred Segmentation')
            axes[i, 2].axis('off')

            # Plot attention map
            avg_attention = np.mean(attention_maps, axis=0)  # Average across structures
            attention_slice = avg_attention[:, :, slice_idx]
            attention_slice = np.rot90(attention_slice)
            axes[i, 3].imshow(img_slice, cmap='gray', alpha=0.7)
            axes[i, 3].imshow(attention_slice, cmap='hot', alpha=0.5)
            axes[i, 3].set_title('Attention Map')
            axes[i, 3].axis('off')

            # Print presence predictions
            print(f"\nSample {i + 1} ({scenario}):")
            print("Structure | True | Pred | Conf")
            print("-" * 35)
            for j, struct_name in enumerate(structure_names):
                true_val = "Present" if true_presence[j] == 1 else "Absent"
                pred_val = "Present" if pred_presence[j] > 0.5 else "Absent"
                conf = pred_presence[j]
                print(f"{struct_name:12} | {true_val:7} | {pred_val:7} | {conf:.3f}")

            # Memory cleanup after each sample
            cleanup_memory()

    plt.tight_layout()

    # Save visualization to Sacred
    fig_path = './temp_visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    ex.add_artifact(fig_path, 'predictions_visualization.png')
    plt.show()

    # Clean up plot
    plt.close(fig)
    cleanup_memory()


@ex.capture
def plot_training_history(history: Dict, logging):
    """Plot training curves and save to Sacred with memory optimization"""
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

    # Presence accuracy
    train_presence_acc = [m['presence_metrics']['overall_accuracy'] for m in history['train_metrics']]
    val_presence_acc = [m['presence_metrics']['overall_accuracy'] for m in history['val_metrics']]

    axes[0, 1].plot(epochs, train_presence_acc, 'b-', label='Train Presence Acc', linewidth=2)
    axes[0, 1].plot(epochs, val_presence_acc, 'r-', label='Val Presence Acc', linewidth=2)
    axes[0, 1].set_title('Presence Detection Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Segmentation Dice
    train_dice = [m['segmentation_metrics']['mean_dice'] for m in history['train_metrics']]
    val_dice = [m['segmentation_metrics']['mean_dice'] for m in history['val_metrics']]

    axes[1, 0].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1, 0].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
    axes[1, 0].set_title('Mean Dice Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save training curves to Sacred
    curves_path = './temp_training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    ex.add_artifact(curves_path, 'training_curves.png')

    if logging['visualize_predictions']:
        plt.show()

    # Clean up
    plt.close(fig)
    cleanup_memory()

    return fig


def analyze_structure_performance(history, structure_names):
    """Analyze per-structure performance and log to Sacred with memory optimization"""
    # Use only the last few metrics to avoid memory issues with full history
    if len(history['train_metrics']) == 0 or len(history['val_metrics']) == 0:
        print("No metrics available for analysis")
        return {}

    final_train_metrics = history['train_metrics'][-1]
    final_val_metrics = history['val_metrics'][-1]

    print("\nFinal Structure-wise Performance Analysis:")
    print("=" * 80)
    print(f"{'Structure':<15} | {'Train Pres':<10} | {'Val Pres':<10} | {'Train Dice':<10} | {'Val Dice':<10}")
    print("-" * 80)

    structure_performance = {}

    # Note: This analysis is simplified for memory-optimized version
    # Structure-specific metrics might not be available due to memory optimization
    overall_train_pres = final_train_metrics['presence_metrics']['overall_accuracy']
    overall_val_pres = final_val_metrics['presence_metrics']['overall_accuracy']
    overall_train_dice = final_train_metrics['segmentation_metrics']['mean_dice']
    overall_val_dice = final_val_metrics['segmentation_metrics']['mean_dice']

    print(
        f"{'Overall':<15} | {overall_train_pres:<10.3f} | {overall_val_pres:<10.3f} | {overall_train_dice:<10.3f} | {overall_val_dice:<10.3f}")

    # Store simplified performance for Sacred logging
    structure_performance['overall'] = {
        'train_presence_acc': overall_train_pres,
        'val_presence_acc': overall_val_pres,
        'train_dice': overall_train_dice,
        'val_dice': overall_val_dice
    }

    # Log overall performance to Sacred
    ex.log_scalar('final.overall.presence_acc', overall_val_pres)
    ex.log_scalar('final.overall.dice', overall_val_dice)

    # Store structure performance in Sacred info
    ex.info['final_structure_performance'] = structure_performance

    print("\nNote: Detailed per-structure analysis limited in memory-optimized mode")

    return structure_performance


@ex.capture
def save_results(history, structure_performance, checkpoint):
    """Save training results with memory optimization"""
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)

    # Save only essential training history to reduce memory usage
    essential_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'learning_rates': history['learning_rates']
    }

    history_path = results_path / 'essential_training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(essential_history, f)

    # Save structure performance analysis
    perf_path = results_path / 'structure_performance.json'
    with open(perf_path, 'w') as f:
        json.dump(structure_performance, f, indent=2)

    # Add results as Sacred artifacts
    ex.add_artifact(str(history_path), 'essential_training_history.pkl')
    ex.add_artifact(str(perf_path), 'structure_performance.json')

    print(f"Results saved to {results_path}")

    # Clean up
    del essential_history
    cleanup_memory()


@ex.automain
# def main(dataset_path, device, seed, logging, sacred_dir, config_path):
def main(dataset_path, device, seed, logging, config_path):
    """Main training pipeline with Sacred integration and memory optimization"""

    # Update Sacred observer directory if provided
    # if sacred_dir != './sacred_logs':
    #     ex.observers.clear()
    #     ex.observers.append(FileStorageObserver(sacred_dir))

    # Load config from file if provided and merge with Sacred config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        print(f"Loaded additional config from: {config_path}")
        # Note: file_config values will override Sacred config when using config_updates

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set device and optimize memory settings
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ex.log_scalar('system.device', str(device))

    # CUDA memory optimization settings
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except:
            pass

        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, structure_names, image_size = create_data_loaders()
    print(f"Loaded dataset with {len(structure_names)} structures")
    print(f"Structure names: {structure_names}")

    # Create model
    print("Creating model...")
    model = create_model(structure_names=structure_names, image_size=image_size, device=device)

    # Log memory usage after model creation
    if torch.cuda.is_available():
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        structure_names=structure_names,
        device=device
    )

    # Train model
    print("Starting training...")
    history = trainer.train()

    # Analyze results
    print("Analyzing results...")
    structure_performance = analyze_structure_performance(history, structure_names)

    # Plot results
    plot_training_history(history)

    # Visualize predictions (with memory optimization)
    if logging['visualize_predictions']:
        print("Visualizing predictions...")
        val_dataset = val_loader.dataset
        visualize_predictions(
            model, val_dataset, structure_names, device=device,
            num_samples=min(logging['num_visualization_samples'], 2)  # Limit samples to save memory
        )

    # Save results
    save_results(history, structure_performance)

    # Final Sacred logging
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_metrics'][-1]['presence_metrics']['overall_accuracy']
    final_val_dice = history['val_metrics'][-1]['segmentation_metrics']['mean_dice']

    ex.log_scalar('final.validation_loss', final_val_loss)
    ex.log_scalar('final.presence_accuracy', final_val_acc)
    ex.log_scalar('final.mean_dice', final_val_dice)

    # Final memory cleanup
    cleanup_memory()

    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    print("Training completed successfully!")

    # Return results for Sacred
    return {
        'final_val_loss': final_val_loss,
        'final_presence_accuracy': final_val_acc,
        'final_mean_dice': final_val_dice,
        'structure_performance': structure_performance
    }


if __name__ == "__main__":
    import sys

    print("Sacred Training Script - Memory Optimized")
    print("Usage examples:")
    print("python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/'")
    print(
        "python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/' sacred_dir='./experiments'")
    print(
        "python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/' config_path='../config/config.json'")
    print(
        "python train_memory_optimized.py with dataset_path='../data/synthetic_medical_dataset/' sacred_dir='./experiments' config_path='../config/config.json'")
    print("\nMemory optimization features enabled:")
    print("- Gradient accumulation support")
    print("- Periodic memory cleanup")
    print("- Limited metric collection")
    print("- Efficient checkpointing")
    print("- GPU memory monitoring")
    print("\nStarting experiment with default parameters if no 'with' clause provided...")

    # If no 'with' arguments provided, run with defaults
    if len(sys.argv) == 1:
        ex.run()
    else:
        # Sacred will handle command line parsing automatically
        pass