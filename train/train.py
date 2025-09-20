import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
import h5py

# Sacred experiment setup
ex = Experiment('sadaan_hdf5')
ex.observers.append(MongoObserver(url='localhost:27017', db_name='sadaan'))

# Import your previous implementations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sadaan import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


@ex.config
def config():
    """Sacred configuration"""
    # Data parameters - now using HDF5 format
    dataset_path = './synthetic_medical_dataset_hdf5'

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
        'gradient_accumulation_steps': 2,
        'memory_cleanup_frequency': 5,
        'max_history_length': 50
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
        'log_frequency': 1,
        'visualize_predictions': True,
        'num_visualization_samples': 3
    }

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42


class HDF5MedicalDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads samples on-demand from HDF5"""

    def __init__(self, data_path: str, split: str = 'train', transform=None):
        """
        Args:
            data_path: Path to the HDF5 dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform function
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Load metadata
        with open(self.data_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.structure_names = self.metadata['structure_names']
        self.image_size = self.metadata['image_size']

        # Open HDF5 file
        self.hdf5_path = self.data_path / f'{self.split}.h5'
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Open file and get basic info
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.n_samples = self.hdf5_file.attrs['n_samples']

        # Preload scenarios for faster access
        self.scenarios = [s.decode('utf-8') if isinstance(s, bytes) else s
                          for s in self.hdf5_file['scenarios'][:]]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")

        # Load image and labels
        image = self.hdf5_file['images'][idx]
        masks = self.hdf5_file['masks'][idx]
        presence_labels = self.hdf5_file['presence_labels'][idx]
        scenario = self.scenarios[idx]

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        masks = torch.from_numpy(masks.astype(np.float32))
        presence_labels = torch.from_numpy(presence_labels.astype(np.int64))

        # Add channel dimension to image
        image = image.unsqueeze(0)  # [1, H, W, D]

        return {
            'image': image,
            'masks': masks,
            'presence_labels': presence_labels,
            'scenario': scenario,
            'index': idx
        }

    def __del__(self):
        """Clean up HDF5 file handle"""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


class AugmentationTransform:
    """Simple augmentation transforms for medical images"""

    def __init__(self, noise_std=5.0, intensity_scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, image):
        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise

        # Scale intensity
        if self.intensity_scale_range != (1.0, 1.0):
            scale_min, scale_max = self.intensity_scale_range
            scale = np.random.uniform(scale_min, scale_max)
            image = image * scale

        # Clip to reasonable range
        image = np.clip(image, 0, 255)

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

        # Overall metrics
        overall_accuracy = (pred_presence_binary == true_presence).float().mean().item()

        return {
            'overall_accuracy': overall_accuracy,
            'mean_structure_accuracy': overall_accuracy
        }


def cleanup_memory():
    """Enhanced memory cleanup"""
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ex.log_scalar('model.total_parameters', total_params)
    ex.log_scalar('model.trainable_parameters', trainable_params)

    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return model


@ex.capture
def create_data_loaders(dataset_path, training, augmentation):
    """Load HDF5 datasets and create data loaders"""

    # Check if dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load metadata
    with open(dataset_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    structure_names = metadata['structure_names']
    image_size = metadata['image_size']

    print(f"Loading HDF5 dataset from {dataset_path}")
    print(f"Structures: {structure_names}")
    print(f"Image size: {image_size}")

    # Create transforms
    train_transform = None
    if augmentation['enabled']:
        train_transform = AugmentationTransform(
            noise_std=augmentation['noise_std'],
            intensity_scale_range=tuple(augmentation['intensity_scale_range'])
        )

    # Create datasets
    train_dataset = HDF5MedicalDataset(dataset_path, split='train', transform=train_transform)
    val_dataset = HDF5MedicalDataset(dataset_path, split='val')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
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

    # Log dataset info to Sacred
    ex.log_scalar('data.train_samples', len(train_dataset))
    ex.log_scalar('data.val_samples', len(val_dataset))
    ex.log_scalar('data.num_structures', len(structure_names))
    ex.info['data.structure_names'] = structure_names
    ex.info['data.image_size'] = image_size

    return train_loader, val_loader, structure_names, image_size


class Trainer:
    """Streamlined training class for HDF5 datasets"""

    @ex.capture
    def __init__(self, model, train_loader, val_loader, structure_names,
                 training, loss_weights, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.structure_names = structure_names
        self.device = device
        self.training_config = training

        # Loss function
        self.loss_fn = SpatialAttentionLoss(
            structure_names,
            weights=loss_weights,
            focal_alpha=0.25,
            focal_gamma=2.0
        )

        # Optimizer
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

        # Training parameters
        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 1)
        self.memory_cleanup_frequency = training.get('memory_cleanup_frequency', 5)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        self.global_step = 0
        self.current_epoch = 0

        # Setup warmup if specified
        warmup_epochs = training.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            self.warmup_scheduler = LearningRateWarmup(
                self.optimizer,
                warmup_epochs,
                training['learning_rate']
            )

    def create_optimizer(self, model, training, optimizer_config):
        """Create optimizer"""
        if optimizer_config['type'] == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=training['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=training['learning_rate']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    def dice_coefficient_fast(self, pred_mask, true_mask, epsilon=1e-6):
        """Fast dice calculation"""
        pred_flat = pred_mask.view(-1)
        true_flat = true_mask.view(-1)
        intersection = (pred_flat * true_flat).sum()
        dice = (2.0 * intersection + epsilon) / (pred_flat.sum() + true_flat.sum() + epsilon)
        return dice

    @ex.capture
    def train_epoch(self, logging) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)

        # Running metrics
        running_presence_correct = 0
        running_presence_total = 0
        running_dice_sum = 0
        running_dice_count = 0

        # Handle warmup
        warmup_epochs = self.training_config.get('warmup_epochs', 0)
        if hasattr(self, 'warmup_scheduler') and self.current_epoch < warmup_epochs:
            self.warmup_scheduler.step()

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            self.global_step += 1

            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            true_masks = batch['masks'].to(self.device, non_blocking=True)
            true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

            # Forward pass
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            outputs = self.model(images)

            # Calculate loss
            targets = {
                'segmentation_targets': true_masks,
                'presence_targets': true_presence
            }
            losses = self.loss_fn(outputs, targets)
            total_loss = losses['total']

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            total_loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.training_config['gradient_clip_norm']
                )
                self.optimizer.step()

            # Store losses
            for key, value in losses.items():
                epoch_losses[key].append(value.item() if hasattr(value, 'item') else float(value))

            # Calculate metrics incrementally
            with torch.no_grad():
                pred_presence_binary = (outputs['presence_probs'] > 0.5).long()
                presence_correct = (pred_presence_binary == true_presence).float().sum()
                running_presence_correct += presence_correct.item()
                running_presence_total += true_presence.numel()

                # Dice for present structures
                for b in range(true_presence.shape[0]):
                    for s in range(true_presence.shape[1]):
                        if true_presence[b, s] == 1:
                            pred_mask = outputs['segmentation_probs'][b, s]
                            true_mask = true_masks[b, s]
                            dice = self.dice_coefficient_fast(pred_mask, true_mask)
                            running_dice_sum += dice.item()
                            running_dice_count += 1

            # Sacred logging
            if self.global_step % logging.get('log_frequency', 1) == 0:
                for key, value in losses.items():
                    ex.log_scalar(f'train.batch.{key}',
                                  value.item() if hasattr(value, 'item') else float(value),
                                  self.global_step)

                ex.log_scalar('train.learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

            # Clean up
            del outputs, losses, total_loss, targets, images, true_masks, true_presence

            if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                cleanup_memory()

            # Update progress bar
            avg_total_loss = np.mean(epoch_losses['total']) if epoch_losses['total'] else 0
            pbar.set_postfix({'loss': f"{avg_total_loss:.4f}"})

        # Calculate final metrics
        presence_accuracy = running_presence_correct / running_presence_total if running_presence_total > 0 else 0
        mean_dice = running_dice_sum / running_dice_count if running_dice_count > 0 else 0

        return {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': {'overall_accuracy': presence_accuracy},
            'segmentation_metrics': {'mean_dice': mean_dice}
        }

    def validate_epoch(self) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)

        running_presence_correct = 0
        running_presence_total = 0
        running_dice_sum = 0
        running_dice_count = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device, non_blocking=True)
                true_masks = batch['masks'].to(self.device, non_blocking=True)
                true_presence = batch['presence_labels'].to(self.device, non_blocking=True)

                outputs = self.model(images)

                targets = {
                    'segmentation_targets': true_masks,
                    'presence_targets': true_presence
                }
                losses = self.loss_fn(outputs, targets)

                for key, value in losses.items():
                    epoch_losses[key].append(value.item() if hasattr(value, 'item') else float(value))

                # Calculate metrics
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

                del outputs, losses, targets, images, true_masks, true_presence

                if (batch_idx + 1) % (self.memory_cleanup_frequency * 2) == 0:
                    cleanup_memory()

                avg_loss = np.mean([loss for losses in epoch_losses.values() for loss in losses])
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        presence_accuracy = running_presence_correct / running_presence_total if running_presence_total > 0 else 0
        mean_dice = running_dice_sum / running_dice_count if running_dice_count > 0 else 0

        return {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': {'overall_accuracy': presence_accuracy},
            'segmentation_metrics': {'mean_dice': mean_dice}
        }

    @ex.capture
    def train(self, training, checkpoint):
        """Main training loop"""
        num_epochs = training['num_epochs']
        early_stopping_patience = training['early_stopping_patience']

        save_path = Path(checkpoint['save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_results = self.train_epoch()

            # Validate
            val_results = self.validate_epoch()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_results['losses']['total'])

            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_results['losses']['total'])
            self.history['val_loss'].append(val_results['losses']['total'])
            self.history['train_metrics'].append(train_results)
            self.history['val_metrics'].append(val_results)
            self.history['learning_rates'].append(current_lr)

            # Sacred logging
            ex.log_scalar('train.epoch.total_loss', train_results['losses']['total'], epoch)
            ex.log_scalar('val.epoch.total_loss', val_results['losses']['total'], epoch)
            ex.log_scalar('train.epoch.presence_accuracy', train_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('val.epoch.presence_accuracy', val_results['presence_metrics']['overall_accuracy'], epoch)
            ex.log_scalar('train.epoch.mean_dice', train_results['segmentation_metrics']['mean_dice'], epoch)
            ex.log_scalar('val.epoch.mean_dice', val_results['segmentation_metrics']['mean_dice'], epoch)

            # Print summary
            print(f"Train Loss: {train_results['losses']['total']:.4f}")
            print(f"Val Loss: {val_results['losses']['total']:.4f}")
            print(f"Train Presence Acc: {train_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Val Presence Acc: {val_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Train Dice: {train_results['segmentation_metrics']['mean_dice']:.4f}")
            print(f"Val Dice: {val_results['segmentation_metrics']['mean_dice']:.4f}")

            # Save checkpoint
            if (epoch + 1) % checkpoint['save_frequency'] == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch)

            # Early stopping
            val_loss = val_results['losses']['total']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_model_path = save_path / 'best_model.pth'
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                print(f"New best model saved! Val loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break

            cleanup_memory()

        print("Training completed!")
        return self.history

    def save_checkpoint(self, filepath: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'structure_names': self.structure_names,
            'is_best': is_best,
            'train_loss_history': self.history['train_loss'][-10:],
            'val_loss_history': self.history['val_loss'][-10:]
        }

        torch.save(checkpoint, filepath)
        ex.add_artifact(str(filepath), f'checkpoint_epoch_{epoch + 1}.pth')


@ex.automain
def main(dataset_path, device, seed, logging):
    """Main training pipeline"""

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, structure_names, image_size = create_data_loaders()

    # Create model
    print("Creating model...")
    model = create_model(structure_names=structure_names, image_size=image_size, device=device)

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

    print("Training completed successfully!")

    # Return results for Sacred
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_metrics'][-1]['presence_metrics']['overall_accuracy']
    final_val_dice = history['val_metrics'][-1]['segmentation_metrics']['mean_dice']

    ex.log_scalar('final.validation_loss', final_val_loss)
    ex.log_scalar('final.presence_accuracy', final_val_acc)
    ex.log_scalar('final.mean_dice', final_val_dice)

    return {
        'final_val_loss': final_val_loss,
        'final_presence_accuracy': final_val_acc,
        'final_mean_dice': final_val_dice
    }


if __name__ == "__main__":
    print("Sacred Training Script - HDF5 Compatible")
    print("Usage examples:")
    print("python train_hdf5.py with dataset_path='./synthetic_medical_dataset_hdf5'")
    print("python train_hdf5.py with dataset_path='./synthetic_medical_dataset_hdf5' training.batch_size=8")
    print("\nFeatures:")
    print("- HDF5 dataset loading for memory efficiency")
    print("- On-demand sample loading")
    print("- Memory optimization")
    print("- Sacred experiment tracking")
    print("\nStarting experiment...")