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
import wandb  # Optional: for experiment tracking
from collections import defaultdict
import pickle


# Import your previous implementations
# from synthetic_data_pipeline import SyntheticDatasetGenerator
# from spatial_attention_model import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


class SyntheticMedicalDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic medical data"""

    def __init__(self, dataset_dict: Dict, transform=None):
        self.images = dataset_dict['images']
        self.masks = dataset_dict['masks']
        self.presence_labels = dataset_dict['presence_labels']
        self.scenarios = dataset_dict['scenarios']
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
            scale = torch.uniform(self.intensity_scale_range[0], self.intensity_scale_range[1])
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


class Trainer:
    """Main training class"""

    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 structure_names: List[str],
                 device: str = 'cuda',
                 batch_size: int = 4,
                 learning_rate: float = 1e-3,
                 num_workers: int = 4):

        self.model = model.to(device)
        self.device = device
        self.structure_names = structure_names
        self.batch_size = batch_size

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )

        # Loss function and optimizer
        self.loss_fn = SpatialAttentionLoss(structure_names)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Metrics calculator
        self.metrics_calc = MetricsCalculator(structure_names)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()

        epoch_losses = defaultdict(list)
        all_pred_presence = []
        all_true_presence = []
        all_pred_masks = []
        all_true_masks = []
        all_true_presence_for_seg = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)  # [B, 1, H, W, D]
            true_masks = batch['masks'].to(self.device)  # [B, num_structures, H, W, D]
            true_presence = batch['presence_labels'].to(self.device)  # [B, num_structures]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Prepare targets
            targets = {
                'segmentation_targets': true_masks,
                'presence_targets': true_presence
            }

            # Calculate loss
            losses = self.loss_fn(outputs, targets)
            total_loss = losses['total']

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Store losses
            for key, value in losses.items():
                epoch_losses[key].append(value.item())

            # Collect predictions for metrics
            with torch.no_grad():
                all_pred_presence.append(outputs['presence_probs'].cpu())
                all_true_presence.append(true_presence.cpu())
                all_pred_masks.append(outputs['segmentation_probs'].cpu())
                all_true_masks.append(true_masks.cpu())
                all_true_presence_for_seg.append(true_presence.cpu())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'seg': f"{losses.get('segmentation', 0):.4f}",
                'abs': f"{losses.get('absence', 0):.4f}"
            })

        # Calculate epoch metrics
        all_pred_presence = torch.cat(all_pred_presence, dim=0)
        all_true_presence = torch.cat(all_true_presence, dim=0)
        all_pred_masks = torch.cat(all_pred_masks, dim=0)
        all_true_masks = torch.cat(all_true_masks, dim=0)
        all_true_presence_for_seg = torch.cat(all_true_presence_for_seg, dim=0)

        # Presence detection metrics
        presence_metrics = self.metrics_calc.calculate_presence_metrics(
            all_pred_presence, all_true_presence
        )

        # Segmentation metrics
        seg_metrics = self.metrics_calc.calculate_segmentation_metrics(
            all_pred_masks, all_true_masks, all_true_presence_for_seg
        )

        # Combine results
        train_results = {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        return train_results

    def validate_epoch(self) -> Dict:
        """Validate for one epoch"""
        self.model.eval()

        epoch_losses = defaultdict(list)
        all_pred_presence = []
        all_true_presence = []
        all_pred_masks = []
        all_true_masks = []
        all_true_presence_for_seg = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                true_masks = batch['masks'].to(self.device)
                true_presence = batch['presence_labels'].to(self.device)

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

                # Collect predictions
                all_pred_presence.append(outputs['presence_probs'].cpu())
                all_true_presence.append(true_presence.cpu())
                all_pred_masks.append(outputs['segmentation_probs'].cpu())
                all_true_masks.append(true_masks.cpu())
                all_true_presence_for_seg.append(true_presence.cpu())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}"
                })

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

        val_results = {
            'losses': {k: np.mean(v) for k, v in epoch_losses.items()},
            'presence_metrics': presence_metrics,
            'segmentation_metrics': seg_metrics
        }

        return val_results

    def train(self, num_epochs: int, save_dir: str = './checkpoints',
              save_frequency: int = 10, early_stopping_patience: int = 20):
        """Main training loop"""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_results = self.train_epoch()

            # Validate
            val_results = self.validate_epoch()

            # Update learning rate scheduler
            val_loss = val_results['losses']['total']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_results['losses']['total'])
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_results)
            self.history['val_metrics'].append(val_results)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"Train Loss: {train_results['losses']['total']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train Presence Acc: {train_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Val Presence Acc: {val_results['presence_metrics']['overall_accuracy']:.4f}")
            print(f"Train Mean Dice: {train_results['segmentation_metrics']['mean_dice']:.4f}")
            print(f"Val Mean Dice: {val_results['segmentation_metrics']['mean_dice']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
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

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break

        print("\nTraining completed!")
        return self.history

    def save_checkpoint(self, filepath: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'structure_names': self.structure_names,
            'is_best': is_best
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


def visualize_predictions(model, dataset, structure_names, device='cuda', num_samples=3):
    """Visualize model predictions on sample data"""
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

            # Get middle slice for visualization
            slice_idx = image.shape[-1] // 2
            img_slice = image[0, 0, :, :, slice_idx].cpu().numpy()

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

    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Presence accuracy
    train_presence_acc = [m['presence_metrics']['overall_accuracy'] for m in history['train_metrics']]
    val_presence_acc = [m['presence_metrics']['overall_accuracy'] for m in history['val_metrics']]

    axes[0, 1].plot(epochs, train_presence_acc, 'b-', label='Train Presence Acc')
    axes[0, 1].plot(epochs, val_presence_acc, 'r-', label='Val Presence Acc')
    axes[0, 1].set_title('Presence Detection Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Segmentation Dice
    train_dice = [m['segmentation_metrics']['mean_dice'] for m in history['train_metrics']]
    val_dice = [m['segmentation_metrics']['mean_dice'] for m in history['val_metrics']]

    axes[1, 0].plot(epochs, train_dice, 'b-', label='Train Dice')
    axes[1, 0].plot(epochs, val_dice, 'r-', label='Val Dice')
    axes[1, 0].set_title('Mean Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# Main training script
def main():
    """Main training pipeline"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    data_generator = SyntheticDatasetGenerator(image_size=(128, 128, 64))
    dataset = data_generator.generate_dataset(
        total_samples=2000,
        validation_split=0.2,
        test_split=0.1,
        position_noise=0.05,
        size_noise=0.2,
        intensity_noise=0.15
    )

    structure_names = dataset['structure_names']
    print(f"Generated dataset with {len(structure_names)} structures")

    # Create PyTorch datasets
    train_transform = AugmentationTransform(noise_std=5.0, intensity_scale_range=(0.9, 1.1))

    train_dataset = SyntheticMedicalDataset(dataset['train'], transform=train_transform)
    val_dataset = SyntheticMedicalDataset(dataset['val'])

    # Create model
    model = SpatialAttentionMedicalSegmenter(
        in_channels=1,
        num_structures=len(structure_names),
        feature_channels=128,  # Reduced for faster training
        spatial_dims=(128, 128, 64)
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        structure_names=structure_names,
        device=device,
        batch_size=2,  # Small batch size for memory
        learning_rate=1e-3,
        num_workers=2
    )

    # Train model
    history = trainer.train(
        num_epochs=100,
        save_dir='./checkpoints',
        save_frequency=10,
        early_stopping_patience=15
    )

    # Plot results
    plot_training_history(history)

    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(model, val_dataset, structure_names, device=device)

    # Save final results
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)

    with open(results_path / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()