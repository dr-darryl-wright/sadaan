import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import h5py
from tqdm import tqdm
import warnings
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# Import your model and dataset classes
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sadaan import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


class HDF5MedicalDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads samples on-demand from HDF5"""

    def __init__(self, data_path: str, split: str = 'test', transform=None):
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


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations"""

    def __init__(self, model, device, structure_names, output_dir='./test_results'):
        self.model = model
        self.device = device
        self.structure_names = structure_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize result storage
        self.results = {
            'predictions': [],
            'targets': [],
            'presence_predictions': [],
            'presence_targets': [],
            'segmentation_predictions': [],
            'segmentation_targets': [],
            'attention_maps': [],
            'position_priors': [],
            'feature_responses': [],
            'raw_segmentation_logits': [],
            'scenarios': [],
            'sample_indices': [],
            'images': [],
            'confidence_scores': []
        }

    def dice_coefficient(self, pred_mask, true_mask, epsilon=1e-6):
        """Calculate Dice coefficient"""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()

        intersection = (pred_flat * true_flat).sum()
        return (2.0 * intersection + epsilon) / (pred_flat.sum() + true_flat.sum() + epsilon)

    def iou_score(self, pred_mask, true_mask, epsilon=1e-6):
        """Calculate Intersection over Union"""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()

        intersection = (pred_flat * true_flat).sum()
        union = pred_flat.sum() + true_flat.sum() - intersection
        return (intersection + epsilon) / (union + epsilon)

    def evaluate_batch(self, batch):
        """Evaluate a single batch and store results"""
        self.model.eval()

        with torch.no_grad():
            images = batch['image'].to(self.device)
            true_masks = batch['masks'].to(self.device)
            true_presence = batch['presence_labels'].to(self.device)
            scenarios = batch['scenario']
            indices = batch['index']

            # Forward pass
            outputs = self.model(images)

            # Store results
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Convert to numpy for storage
                image_np = images[i].cpu().numpy()
                true_masks_np = true_masks[i].cpu().numpy()
                true_presence_np = true_presence[i].cpu().numpy()

                pred_masks_np = outputs['segmentation_probs'][i].cpu().numpy()
                pred_presence_np = outputs['presence_probs'][i].cpu().numpy()

                # Store attention maps and anatomical priors
                attention_maps_np = outputs['attention_maps'][i].cpu().numpy()
                position_priors_np = outputs['position_priors'][i].cpu().numpy()
                feature_responses_np = outputs['feature_responses'][i].cpu().numpy()
                raw_seg_logits_np = outputs['raw_logits'][i].cpu().numpy()

                # Calculate confidence scores (entropy-based)
                presence_probs = outputs['presence_probs'][i].cpu().numpy()
                confidence = 1.0 - (-presence_probs * np.log(presence_probs + 1e-8) -
                                    (1 - presence_probs) * np.log(1 - presence_probs + 1e-8))

                self.results['images'].append(image_np)
                self.results['segmentation_targets'].append(true_masks_np)
                self.results['presence_targets'].append(true_presence_np)
                self.results['segmentation_predictions'].append(pred_masks_np)
                self.results['presence_predictions'].append(pred_presence_np)
                self.results['attention_maps'].append(attention_maps_np)
                self.results['position_priors'].append(position_priors_np)
                self.results['feature_responses'].append(feature_responses_np)
                self.results['raw_segmentation_logits'].append(raw_seg_logits_np)
                self.results['scenarios'].append(scenarios[i])
                self.results['sample_indices'].append(indices[i].item())
                self.results['confidence_scores'].append(confidence)

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        print("Calculating evaluation metrics...")

        # Convert to arrays
        presence_targets = np.array(self.results['presence_targets'])  # [N, C]
        presence_predictions = np.array(self.results['presence_predictions'])  # [N, C]
        segmentation_targets = np.array(self.results['segmentation_targets'])  # [N, C, H, W, D]
        segmentation_predictions = np.array(self.results['segmentation_predictions'])  # [N, C, H, W, D]
        scenarios = self.results['scenarios']
        confidence_scores = np.array(self.results['confidence_scores'])  # [N, C]

        metrics = {}

        # ============ PRESENCE DETECTION METRICS ============
        print("  Computing presence detection metrics...")
        presence_pred_binary = (presence_predictions > 0.5).astype(int)

        # Overall accuracy
        overall_accuracy = (presence_pred_binary == presence_targets).mean()
        metrics['presence_overall_accuracy'] = float(overall_accuracy)

        # Per-structure metrics
        presence_metrics = {}
        for i, struct_name in enumerate(self.structure_names):
            # Get predictions and targets for this structure
            y_true = presence_targets[:, i]
            y_pred = presence_pred_binary[:, i]
            y_score = presence_predictions[:, i]

            # Basic metrics
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # AUC-ROC and AUC-PR
            try:
                auc_roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
                auc_pr = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else y_true.mean()
            except:
                auc_roc = 0.5
                auc_pr = y_true.mean() if len(y_true) > 0 else 0.5

            presence_metrics[struct_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'support_positive': int(y_true.sum()),
                'support_negative': int((1 - y_true).sum())
            }

        metrics['presence_per_structure'] = presence_metrics

        # ============ SEGMENTATION METRICS ============
        print("  Computing segmentation metrics...")
        segmentation_pred_binary = (segmentation_predictions > 0.5).astype(float)

        segmentation_metrics = {}
        for i, struct_name in enumerate(self.structure_names):
            dice_scores = []
            iou_scores = []

            # Only calculate for samples where structure is present
            present_mask = presence_targets[:, i] == 1
            n_present = present_mask.sum()

            if n_present > 0:
                for j in range(len(segmentation_targets)):
                    if presence_targets[j, i] == 1:
                        true_mask = segmentation_targets[j, i]
                        pred_mask = segmentation_pred_binary[j, i]

                        dice = self.dice_coefficient(pred_mask, true_mask)
                        iou = self.iou_score(pred_mask, true_mask)

                        dice_scores.append(dice)
                        iou_scores.append(iou)

                segmentation_metrics[struct_name] = {
                    'mean_dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
                    'std_dice': float(np.std(dice_scores)) if dice_scores else 0.0,
                    'mean_iou': float(np.mean(iou_scores)) if iou_scores else 0.0,
                    'std_iou': float(np.std(iou_scores)) if iou_scores else 0.0,
                    'num_samples': len(dice_scores)
                }
            else:
                segmentation_metrics[struct_name] = {
                    'mean_dice': 0.0,
                    'std_dice': 0.0,
                    'mean_iou': 0.0,
                    'std_iou': 0.0,
                    'num_samples': 0
                }

        metrics['segmentation_per_structure'] = segmentation_metrics

        # Overall segmentation metrics
        all_dice = [m['mean_dice'] for m in segmentation_metrics.values() if m['num_samples'] > 0]
        all_iou = [m['mean_iou'] for m in segmentation_metrics.values() if m['num_samples'] > 0]

        metrics['segmentation_overall'] = {
            'mean_dice': float(np.mean(all_dice)) if all_dice else 0.0,
            'mean_iou': float(np.mean(all_iou)) if all_iou else 0.0
        }

        # ============ SCENARIO-BASED METRICS ============
        print("  Computing scenario-based metrics...")
        scenario_metrics = {}
        unique_scenarios = list(set(scenarios))

        for scenario in unique_scenarios:
            scenario_indices = [i for i, s in enumerate(scenarios) if s == scenario]
            if not scenario_indices:
                continue

            scenario_presence_targets = presence_targets[scenario_indices]
            scenario_presence_preds = presence_pred_binary[scenario_indices]

            scenario_accuracy = (scenario_presence_preds == scenario_presence_targets).mean()

            scenario_metrics[scenario] = {
                'accuracy': float(scenario_accuracy),
                'num_samples': len(scenario_indices),
                'avg_structures_present': float(scenario_presence_targets.mean()),
                'avg_structures_predicted': float(scenario_presence_preds.mean())
            }

        metrics['scenario_based'] = scenario_metrics

        # ============ CONFIDENCE ANALYSIS ============
        print("  Computing confidence analysis...")
        # Calculate calibration metrics
        confidence_bins = np.linspace(0, 1, 11)
        calibration_data = []

        for i in range(len(self.structure_names)):
            conf = confidence_scores[:, i]
            acc = (presence_pred_binary[:, i] == presence_targets[:, i]).astype(float)

            for j in range(len(confidence_bins) - 1):
                bin_mask = (conf >= confidence_bins[j]) & (conf < confidence_bins[j + 1])
                if bin_mask.sum() > 0:
                    bin_conf = conf[bin_mask].mean()
                    bin_acc = acc[bin_mask].mean()
                    calibration_data.append({
                        'structure': self.structure_names[i],
                        'bin_start': confidence_bins[j],
                        'bin_end': confidence_bins[j + 1],
                        'confidence': bin_conf,
                        'accuracy': bin_acc,
                        'count': bin_mask.sum()
                    })

        metrics['calibration_data'] = calibration_data

        print("  Metrics calculation completed!")
        return metrics

    def create_visualizations(self, metrics, max_samples=50):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # ============ NEW: DETAILED SAMPLE ANALYSIS ============
        print("  Creating detailed sample analysis...")
        self.visualize_detailed_samples(max_samples=min(10, max_samples))

        # ============ NEW: ATTENTION MAPS VISUALIZATION ============
        print("  Creating attention maps visualization...")
        self.visualize_attention_maps(max_samples=min(8, max_samples))

        # ============ NEW: ANATOMICAL PRIORS VISUALIZATION ============
        print("  Creating anatomical priors visualization...")
        self.visualize_anatomical_priors()

        # ============ NEW: SEGMENTATION COMPARISON ============
        print("  Creating segmentation comparisons...")
        self.visualize_segmentation_comparison(max_samples=min(6, max_samples))

        # ============ SAMPLE VISUALIZATIONS ============
        print("  Creating sample visualizations...")
        self.visualize_samples(max_samples)

        # ============ PRESENCE DETECTION VISUALIZATIONS ============
        print("  Creating presence detection visualizations...")
        self.visualize_presence_metrics(metrics)

        # ============ SEGMENTATION VISUALIZATIONS ============
        print("  Creating segmentation visualizations...")
        self.visualize_segmentation_metrics(metrics)

        # ============ SCENARIO ANALYSIS ============
        print("  Creating scenario analysis...")
        self.visualize_scenario_analysis(metrics)

        # ============ CONFIDENCE ANALYSIS ============
        print("  Creating confidence analysis...")
        self.visualize_confidence_analysis(metrics)

        # ============ CONFUSION MATRICES ============
        print("  Creating confusion matrices...")
        self.visualize_confusion_matrices()

        # ============ ROC AND PR CURVES ============
        print("  Creating ROC and PR curves...")
        self.visualize_roc_pr_curves(metrics)

        print("  Visualizations completed!")

    def visualize_detailed_samples(self, max_samples=10):
        """Create detailed visualization of individual samples with all components"""
        n_samples = min(max_samples, len(self.results['images']))

        for sample_idx in range(n_samples):
            # Get sample data
            image = self.results['images'][sample_idx][0]  # Remove channel dim
            scenario = self.results['scenarios'][sample_idx]
            presence_true = self.results['presence_targets'][sample_idx]
            presence_pred = self.results['presence_predictions'][sample_idx]
            attention_maps = self.results['attention_maps'][sample_idx]
            position_priors = self.results['position_priors'][sample_idx]

            # Get middle slices
            d_mid = image.shape[-1] // 2
            h_mid = image.shape[0] // 2
            w_mid = image.shape[1] // 2

            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 15))

            # Main image (axial, coronal, sagittal)
            ax1 = plt.subplot(3, 6, 1)
            img_axial = np.rot90(image[..., d_mid], -1)
            ax1.imshow(img_axial, cmap='gray', vmin=0, vmax=255)
            ax1.set_title(f'Original (Axial)\n{scenario}')
            ax1.axis('off')

            ax2 = plt.subplot(3, 6, 7)
            img_coronal = np.rot90(image[h_mid, ...], -1)
            ax2.imshow(img_coronal, cmap='gray', vmin=0, vmax=255)
            ax2.set_title('Original (Coronal)')
            ax2.axis('off')

            ax3 = plt.subplot(3, 6, 13)
            img_sagittal = np.rot90(image[:, w_mid, :], -1)
            ax3.imshow(img_sagittal, cmap='gray', vmin=0, vmax=255)
            ax3.set_title('Original (Sagittal)')
            ax3.axis('off')

            # Find most interesting structures (present ones with good predictions)
            presence_pred_binary = (presence_pred > 0.5).astype(int)
            interesting_structures = []
            for i, struct_name in enumerate(self.structure_names):
                if presence_true[i] == 1:  # Structure is present
                    interesting_structures.append((i, struct_name))

            # Limit to top 4 structures for visualization
            interesting_structures = interesting_structures[:4]

            # Visualize attention maps and priors for interesting structures
            for plot_idx, (struct_idx, struct_name) in enumerate(interesting_structures):
                col = plot_idx + 2

                # Attention map (axial)
                ax = plt.subplot(3, 6, col)
                attention_slice = attention_maps[struct_idx, ..., d_mid]
                im = ax.imshow(np.rot90(attention_slice, -1), cmap='hot', alpha=0.7, vmin=0, vmax=1)
                ax.imshow(img_axial, cmap='gray', alpha=0.3, vmin=0, vmax=255)
                ax.set_title(f'{struct_name[:8]}\nAttention (Axial)')
                ax.axis('off')

                # Position prior (coronal)
                ax = plt.subplot(3, 6, col + 6)
                prior_slice = position_priors[struct_idx, h_mid, ...]
                ax.imshow(np.rot90(prior_slice, -1), cmap='viridis', alpha=0.7, vmin=-1, vmax=1)
                ax.imshow(img_coronal, cmap='gray', alpha=0.3, vmin=0, vmax=255)
                ax.set_title(f'{struct_name[:8]}\nPrior (Coronal)')
                ax.axis('off')

                # Combined view (sagittal)
                ax = plt.subplot(3, 6, col + 12)
                combined = attention_maps[struct_idx, :, w_mid, :] + position_priors[struct_idx, :, w_mid, :]
                ax.imshow(np.rot90(combined, -1), cmap='plasma', alpha=0.7)
                ax.imshow(img_sagittal, cmap='gray', alpha=0.3, vmin=0, vmax=255)
                ax.set_title(f'{struct_name[:8]}\nCombined (Sagittal)')
                ax.axis('off')

            plt.suptitle(f'Sample {sample_idx + 1}: Detailed Analysis\nScenario: {scenario}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'detailed_sample_{sample_idx + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def visualize_attention_maps(self, max_samples=8):
        """Visualize attention maps for different structures"""
        n_samples = min(max_samples, len(self.results['images']))
        n_structures = len(self.structure_names)

        # Create a comprehensive attention visualization
        fig, axes = plt.subplots(n_samples, n_structures, figsize=(3 * n_structures, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for sample_idx in range(n_samples):
            image = self.results['images'][sample_idx][0]
            attention_maps = self.results['attention_maps'][sample_idx]
            presence_true = self.results['presence_targets'][sample_idx]

            # Get middle slice
            d_mid = image.shape[-1] // 2
            img_slice = np.rot90(image[..., d_mid], -1)

            for struct_idx in range(n_structures):
                ax = axes[sample_idx, struct_idx]

                # Show attention map overlaid on image
                attention_slice = attention_maps[struct_idx, ..., d_mid]
                attention_slice = np.rot90(attention_slice, -1)

                # Background image
                ax.imshow(img_slice, cmap='gray', alpha=0.6, vmin=0, vmax=255)

                # Attention overlay
                im = ax.imshow(attention_slice, cmap='hot', alpha=0.7, vmin=0, vmax=1)

                # Title with presence info
                is_present = "✓" if presence_true[struct_idx] == 1 else "✗"
                ax.set_title(f'{self.structure_names[struct_idx][:8]} {is_present}', fontsize=10)
                ax.axis('off')

                # Add colorbar for the first row
                if sample_idx == 0:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

        plt.suptitle('Attention Maps by Structure and Sample\n(✓=Present, ✗=Absent)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_maps_overview.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_anatomical_priors(self):
        """Visualize learned anatomical position priors"""
        # Get the first sample to extract priors (they should be the same for all samples)
        if len(self.results['position_priors']) == 0:
            print("    No position priors available for visualization")
            return

        position_priors = self.results['position_priors'][0]  # [num_structures, D, H, W]
        n_structures = len(self.structure_names)

        # Create visualization showing priors in different views
        fig, axes = plt.subplots(3, n_structures, figsize=(3 * n_structures, 12))
        if n_structures == 1:
            axes = axes.reshape(-1, 1)

        for struct_idx in range(n_structures):
            prior = position_priors[struct_idx]  # [D, H, W]

            # Get middle slices
            d_mid = prior.shape[0] // 2
            h_mid = prior.shape[1] // 2
            w_mid = prior.shape[2] // 2

            # Axial view
            ax = axes[0, struct_idx]
            im1 = ax.imshow(np.rot90(prior[d_mid, ...], -1), cmap='coolwarm', vmin=-2, vmax=2)
            ax.set_title(f'{self.structure_names[struct_idx]}\nAxial Prior')
            ax.axis('off')
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

            # Coronal view
            ax = axes[1, struct_idx]
            im2 = ax.imshow(np.rot90(prior[:, h_mid, :], -1), cmap='coolwarm', vmin=-2, vmax=2)
            ax.set_title('Coronal Prior')
            ax.axis('off')
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

            # Sagittal view
            ax = axes[2, struct_idx]
            im3 = ax.imshow(np.rot90(prior[:, :, w_mid], -1), cmap='coolwarm', vmin=-2, vmax=2)
            ax.set_title('Sagittal Prior')
            ax.axis('off')
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('Learned Anatomical Position Priors\n(Red=High Prior, Blue=Low Prior)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anatomical_priors.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_segmentation_comparison(self, max_samples=6):
        """Compare true segmentations with predictions"""
        n_samples = min(max_samples, len(self.results['images']))

        for sample_idx in range(n_samples):
            image = self.results['images'][sample_idx][0]
            true_masks = self.results['segmentation_targets'][sample_idx]
            pred_masks = self.results['segmentation_predictions'][sample_idx]
            presence_true = self.results['presence_targets'][sample_idx]
            scenario = self.results['scenarios'][sample_idx]

            # Find present structures
            present_structures = [(i, name) for i, name in enumerate(self.structure_names)
                                  if presence_true[i] == 1]

            if not present_structures:
                continue

            n_present = len(present_structures)
            fig, axes = plt.subplots(3, n_present, figsize=(4 * n_present, 12))
            if n_present == 1:
                axes = axes.reshape(-1, 1)

            # Get middle slice
            d_mid = image.shape[-1] // 2
            img_slice = np.rot90(image[..., d_mid], -1)

            for col, (struct_idx, struct_name) in enumerate(present_structures):
                true_mask = true_masks[struct_idx, ..., d_mid]
                pred_mask = pred_masks[struct_idx, ..., d_mid]

                true_mask = np.rot90(true_mask, -1)
                pred_mask = np.rot90(pred_mask, -1)

                # Original with true mask overlay
                ax = axes[0, col]
                ax.imshow(img_slice, cmap='gray', alpha=0.7, vmin=0, vmax=255)
                ax.imshow(true_mask, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f'{struct_name}\nTrue Segmentation')
                ax.axis('off')

                # Original with predicted mask overlay
                ax = axes[1, col]
                ax.imshow(img_slice, cmap='gray', alpha=0.7, vmin=0, vmax=255)
                ax.imshow(pred_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f'{struct_name}\nPredicted Segmentation')
                ax.axis('off')

                # Difference map (TP=white, FP=red, FN=blue, TN=black)
                ax = axes[2, col]
                true_binary = (true_mask > 0.5).astype(int)
                pred_binary = (pred_mask > 0.5).astype(int)

                diff_map = np.zeros((*true_binary.shape, 3))  # RGB

                # True Positives (white)
                tp_mask = (true_binary == 1) & (pred_binary == 1)
                diff_map[tp_mask] = [1, 1, 1]

                # False Positives (red)
                fp_mask = (true_binary == 0) & (pred_binary == 1)
                diff_map[fp_mask] = [1, 0, 0]

                # False Negatives (blue)
                fn_mask = (true_binary == 1) & (pred_binary == 0)
                diff_map[fn_mask] = [0, 0, 1]

                # True Negatives (black) - already initialized as [0,0,0]

                ax.imshow(diff_map)

                # Calculate metrics for this structure
                dice = self.dice_coefficient(pred_binary, true_binary)
                iou = self.iou_score(pred_binary, true_binary)

                ax.set_title(f'{struct_name}\nDice: {dice:.3f}, IoU: {iou:.3f}\nTP=White, FP=Red, FN=Blue')
                ax.axis('off')

            plt.suptitle(f'Sample {sample_idx + 1}: Segmentation Comparison\nScenario: {scenario}', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'segmentation_comparison_{sample_idx + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def visualize_samples(self, max_samples=50):
        """Visualize individual test samples with predictions"""
        n_samples = min(max_samples, len(self.results['images']))
        samples_per_figure = 6
        n_figures = (n_samples + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(n_figures):
            start_idx = fig_idx * samples_per_figure
            end_idx = min(start_idx + samples_per_figure, n_samples)
            current_samples = end_idx - start_idx

            fig, axes = plt.subplots(current_samples, 4, figsize=(20, 5 * current_samples))
            if current_samples == 1:
                axes = axes.reshape(1, -1)

            for sample_idx in range(current_samples):
                idx = start_idx + sample_idx

                # Get sample data
                image = self.results['images'][idx][0]  # Remove channel dim
                scenario = self.results['scenarios'][idx]
                presence_true = self.results['presence_targets'][idx]
                presence_pred = self.results['presence_predictions'][idx]

                # Get middle slice
                middle_slice = image.shape[-1] // 2
                img_slice = image[..., middle_slice]
                img_slice = np.rot90(img_slice, -1)

                # Original image
                axes[sample_idx, 0].imshow(img_slice, cmap='gray', vmin=0, vmax=255)
                axes[sample_idx, 0].set_title(f'Original\nScenario: {scenario}')
                axes[sample_idx, 0].axis('off')

                # True presence labels
                present_true = [self.structure_names[i] for i, p in enumerate(presence_true) if p == 1]
                absent_true = [self.structure_names[i] for i, p in enumerate(presence_true) if p == 0]

                text_true = f"Present ({len(present_true)}):\n"
                text_true += "\n".join(present_true[:8])  # Limit to first 8
                if len(present_true) > 8:
                    text_true += f"\n...and {len(present_true) - 8} more"

                axes[sample_idx, 1].text(0.05, 0.95, text_true,
                                         transform=axes[sample_idx, 1].transAxes,
                                         fontsize=8, verticalalignment='top', color='green')
                axes[sample_idx, 1].set_title('True Labels')
                axes[sample_idx, 1].axis('off')

                # Predicted presence labels
                presence_pred_binary = (presence_pred > 0.5).astype(int)
                present_pred = [self.structure_names[i] for i, p in enumerate(presence_pred_binary) if p == 1]

                text_pred = f"Predicted ({len(present_pred)}):\n"
                text_pred += "\n".join(present_pred[:8])
                if len(present_pred) > 8:
                    text_pred += f"\n...and {len(present_pred) - 8} more"

                axes[sample_idx, 2].text(0.05, 0.95, text_pred,
                                         transform=axes[sample_idx, 2].transAxes,
                                         fontsize=8, verticalalignment='top', color='blue')
                axes[sample_idx, 2].set_title('Predictions')
                axes[sample_idx, 2].axis('off')

                # Prediction confidence heatmap
                conf_scores = self.results['confidence_scores'][idx]
                correct_predictions = (presence_pred_binary == presence_true).astype(float)

                # Create a simple heatmap showing confidence and correctness
                y_pos = np.arange(len(self.structure_names))
                colors = ['red' if c == 0 else 'green' for c in correct_predictions]

                bars = axes[sample_idx, 3].barh(y_pos, conf_scores, color=colors, alpha=0.7)
                axes[sample_idx, 3].set_yticks(y_pos)
                axes[sample_idx, 3].set_yticklabels([name[:8] for name in self.structure_names], fontsize=6)
                axes[sample_idx, 3].set_xlabel('Confidence')
                axes[sample_idx, 3].set_title('Confidence & Correctness\n(Green=Correct, Red=Wrong)')
                axes[sample_idx, 3].set_xlim(0, 1)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'sample_predictions_{fig_idx + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def visualize_presence_metrics(self, metrics):
        """Visualize presence detection performance"""
        presence_metrics = metrics['presence_per_structure']

        # Extract metrics for plotting
        structures = list(presence_metrics.keys())
        f1_scores = [presence_metrics[s]['f1_score'] for s in structures]
        precisions = [presence_metrics[s]['precision'] for s in structures]
        recalls = [presence_metrics[s]['recall'] for s in structures]
        auc_rocs = [presence_metrics[s]['auc_roc'] for s in structures]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # F1 Scores
        bars1 = axes[0, 0].bar(range(len(structures)), f1_scores, color='skyblue')
        axes[0, 0].set_title('F1 Scores by Structure')
        axes[0, 0].set_xlabel('Anatomical Structures')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_xticks(range(len(structures)))
        axes[0, 0].set_xticklabels(structures, rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1)

        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Precision vs Recall
        axes[0, 1].scatter(precisions, recalls, s=100, alpha=0.7, c='coral')
        for i, struct in enumerate(structures):
            axes[0, 1].annotate(struct[:8], (precisions[i], recalls[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)

        # AUC-ROC scores
        bars3 = axes[1, 0].bar(range(len(structures)), auc_rocs, color='lightgreen')
        axes[1, 0].set_title('AUC-ROC Scores by Structure')
        axes[1, 0].set_xlabel('Anatomical Structures')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_xticks(range(len(structures)))
        axes[1, 0].set_xticklabels(structures, rotation=45, ha='right')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()

        # Support (number of positive samples)
        supports = [presence_metrics[s]['support_positive'] for s in structures]
        bars4 = axes[1, 1].bar(range(len(structures)), supports, color='gold')
        axes[1, 1].set_title('Support (Number of Positive Samples)')
        axes[1, 1].set_xlabel('Anatomical Structures')
        axes[1, 1].set_ylabel('Number of Positive Samples')
        axes[1, 1].set_xticks(range(len(structures)))
        axes[1, 1].set_xticklabels(structures, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'presence_detection_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_segmentation_metrics(self, metrics):
        """Visualize segmentation performance"""
        seg_metrics = metrics['segmentation_per_structure']

        structures = list(seg_metrics.keys())
        dice_means = [seg_metrics[s]['mean_dice'] for s in structures]
        dice_stds = [seg_metrics[s]['std_dice'] for s in structures]
        iou_means = [seg_metrics[s]['mean_iou'] for s in structures]
        iou_stds = [seg_metrics[s]['std_iou'] for s in structures]
        num_samples = [seg_metrics[s]['num_samples'] for s in structures]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Dice scores with error bars
        x_pos = np.arange(len(structures))
        bars1 = axes[0].bar(x_pos, dice_means, yerr=dice_stds, capsize=5, color='lightblue', alpha=0.8)
        axes[0].set_title('Dice Scores by Structure')
        axes[0].set_xlabel('Anatomical Structures')
        axes[0].set_ylabel('Dice Score')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(structures, rotation=45, ha='right')
        axes[0].set_ylim(0, 1)

        # IoU scores with error bars
        bars2 = axes[1].bar(x_pos, iou_means, yerr=iou_stds, capsize=5, color='lightgreen', alpha=0.8)
        axes[1].set_title('IoU Scores by Structure')
        axes[1].set_xlabel('Anatomical Structures')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(structures, rotation=45, ha='right')
        axes[1].set_ylim(0, 1)

        # Number of samples used for evaluation
        bars3 = axes[2].bar(x_pos, num_samples, color='orange', alpha=0.8)
        axes[2].set_title('Number of Samples with Present Structure')
        axes[2].set_xlabel('Anatomical Structures')
        axes[2].set_ylabel('Number of Samples')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(structures, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'segmentation_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_scenario_analysis(self, metrics):
        """Visualize performance by scenario"""
        scenario_metrics = metrics['scenario_based']

        scenarios = list(scenario_metrics.keys())
        accuracies = [scenario_metrics[s]['accuracy'] for s in scenarios]
        num_samples = [scenario_metrics[s]['num_samples'] for s in scenarios]
        avg_present = [scenario_metrics[s]['avg_structures_present'] for s in scenarios]
        avg_predicted = [scenario_metrics[s]['avg_structures_predicted'] for s in scenarios]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy by scenario
        bars1 = axes[0, 0].bar(range(len(scenarios)), accuracies, color='lightblue')
        axes[0, 0].set_title('Accuracy by Scenario')
        axes[0, 0].set_xlabel('Scenarios')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(scenarios)))
        axes[0, 0].set_xticklabels([s[:15] for s in scenarios], rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1)

        # Sample count by scenario
        bars2 = axes[0, 1].bar(range(len(scenarios)), num_samples, color='lightcoral')
        axes[0, 1].set_title('Sample Count by Scenario')
        axes[0, 1].set_xlabel('Scenarios')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_xticks(range(len(scenarios)))
        axes[0, 1].set_xticklabels([s[:15] for s in scenarios], rotation=45, ha='right')

        # True vs Predicted structure count
        x_pos = np.arange(len(scenarios))
        width = 0.35

        axes[1, 0].bar(x_pos - width / 2, avg_present, width, label='True', color='green', alpha=0.7)
        axes[1, 0].bar(x_pos + width / 2, avg_predicted, width, label='Predicted', color='blue', alpha=0.7)
        axes[1, 0].set_title('Average Number of Structures: True vs Predicted')
        axes[1, 0].set_xlabel('Scenarios')
        axes[1, 0].set_ylabel('Average Number of Structures')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([s[:15] for s in scenarios], rotation=45, ha='right')
        axes[1, 0].legend()

        # Scatter plot: complexity vs accuracy
        axes[1, 1].scatter(avg_present, accuracies, s=[n * 5 for n in num_samples], alpha=0.7, c='purple')
        for i, scenario in enumerate(scenarios):
            axes[1, 1].annotate(scenario[:8], (avg_present[i], accuracies[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Average Number of Present Structures')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Scenario Complexity\n(Bubble size = sample count)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_confidence_analysis(self, metrics):
        """Visualize model confidence and calibration"""
        calibration_data = metrics['calibration_data']

        if not calibration_data:
            print("    No calibration data available, skipping confidence analysis")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(calibration_data)

        # Overall calibration plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Calibration curve
        try:
            avg_conf_by_bin = df.groupby('bin_start')['confidence'].mean()
            avg_acc_by_bin = df.groupby('bin_start')['accuracy'].mean()

            # Ensure we have valid data
            if len(avg_conf_by_bin) > 0 and len(avg_acc_by_bin) > 0:
                axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
                axes[0, 0].plot(avg_conf_by_bin.values, avg_acc_by_bin.values, 'o-',
                                color='red', label='Model calibration')
                axes[0, 0].set_xlabel('Mean Predicted Probability')
                axes[0, 0].set_ylabel('Fraction of Positives')
                axes[0, 0].set_title('Reliability Diagram (Calibration Curve)')
                axes[0, 0].legend()
                axes[0, 0].set_xlim(0, 1)
                axes[0, 0].set_ylim(0, 1)
            else:
                axes[0, 0].text(0.5, 0.5, 'No calibration data available',
                                ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Reliability Diagram (Calibration Curve)')
        except Exception as e:
            print(f"    Warning: Could not create calibration curve: {e}")
            axes[0, 0].text(0.5, 0.5, 'Error creating calibration curve',
                            ha='center', va='center', transform=axes[0, 0].transAxes)

        # Confidence distribution
        try:
            all_confidences = []
            for conf_array in self.results['confidence_scores']:
                if isinstance(conf_array, np.ndarray):
                    all_confidences.extend(conf_array.flatten())
                else:
                    all_confidences.extend(np.array(conf_array).flatten())

            if len(all_confidences) > 0:
                axes[0, 1].hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Confidence Scores')
            else:
                axes[0, 1].text(0.5, 0.5, 'No confidence data available',
                                ha='center', va='center', transform=axes[0, 1].transAxes)
        except Exception as e:
            print(f"    Warning: Could not create confidence distribution: {e}")
            axes[0, 1].text(0.5, 0.5, 'Error creating confidence distribution',
                            ha='center', va='center', transform=axes[0, 1].transAxes)

        # Per-structure confidence vs accuracy
        try:
            structure_conf_acc = {}
            for struct in self.structure_names:
                struct_data = df[df['structure'] == struct]
                if not struct_data.empty:
                    structure_conf_acc[struct] = {
                        'confidence': struct_data['confidence'].mean(),
                        'accuracy': struct_data['accuracy'].mean()
                    }

            if structure_conf_acc:
                conf_vals = [structure_conf_acc[s]['confidence'] for s in structure_conf_acc.keys()]
                acc_vals = [structure_conf_acc[s]['accuracy'] for s in structure_conf_acc.keys()]

                axes[1, 0].scatter(conf_vals, acc_vals, s=100, alpha=0.7, c='orange')
                for i, struct in enumerate(structure_conf_acc.keys()):
                    if i < len(conf_vals) and i < len(acc_vals):
                        axes[1, 0].annotate(struct[:8], (conf_vals[i], acc_vals[i]),
                                            xytext=(5, 5), textcoords='offset points', fontsize=8)
                axes[1, 0].set_xlabel('Average Confidence')
                axes[1, 0].set_ylabel('Average Accuracy')
                axes[1, 0].set_title('Confidence vs Accuracy by Structure')
                axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            else:
                axes[1, 0].text(0.5, 0.5, 'No per-structure data available',
                                ha='center', va='center', transform=axes[1, 0].transAxes)
        except Exception as e:
            print(f"    Warning: Could not create confidence vs accuracy plot: {e}")
            axes[1, 0].text(0.5, 0.5, 'Error creating confidence vs accuracy plot',
                            ha='center', va='center', transform=axes[1, 0].transAxes)

        # Calibration error by structure
        try:
            ece_by_structure = {}  # Expected Calibration Error
            for struct in self.structure_names:
                struct_data = df[df['structure'] == struct]
                if not struct_data.empty and struct_data['count'].sum() > 0:
                    ece = 0
                    total_samples = struct_data['count'].sum()
                    for _, row in struct_data.iterrows():
                        if total_samples > 0:
                            bin_weight = row['count'] / total_samples
                            ece += bin_weight * abs(row['confidence'] - row['accuracy'])
                    ece_by_structure[struct] = ece

            if ece_by_structure:
                structures = list(ece_by_structure.keys())
                ece_values = list(ece_by_structure.values())

                # Create colors list with proper length
                colors = plt.cm.Set3(np.linspace(0, 1, len(structures)))

                bars = axes[1, 1].bar(range(len(structures)), ece_values, color=colors)
                axes[1, 1].set_title('Expected Calibration Error by Structure')
                axes[1, 1].set_xlabel('Anatomical Structures')
                axes[1, 1].set_ylabel('Expected Calibration Error')
                axes[1, 1].set_xticks(range(len(structures)))
                axes[1, 1].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
            else:
                axes[1, 1].text(0.5, 0.5, 'No calibration error data available',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
        except Exception as e:
            print(f"    Warning: Could not create calibration error plot: {e}")
            axes[1, 1].text(0.5, 0.5, 'Error creating calibration error plot',
                            ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_confusion_matrices(self):
        """Create confusion matrices for each structure"""
        presence_targets = np.array(self.results['presence_targets'])
        presence_predictions = np.array(self.results['presence_predictions'])
        presence_pred_binary = (presence_predictions > 0.5).astype(int)

        n_structures = len(self.structure_names)
        cols = 4
        rows = (n_structures + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, struct_name in enumerate(self.structure_names):
            row = i // cols
            col = i % cols

            y_true = presence_targets[:, i]
            y_pred = presence_pred_binary[:, i]

            cm = confusion_matrix(y_true, y_pred)

            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Absent', 'Present'],
                        yticklabels=['Absent', 'Present'],
                        ax=axes[row, col])

            axes[row, col].set_title(f'{struct_name}')
            axes[row, col].set_ylabel('True Label')
            axes[row, col].set_xlabel('Predicted Label')

        # Hide empty subplots
        for i in range(n_structures, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_roc_pr_curves(self, metrics):
        """Create ROC and PR curves"""
        presence_targets = np.array(self.results['presence_targets'])
        presence_predictions = np.array(self.results['presence_predictions'])

        # Select a subset of structures for cleaner plots
        n_structures_to_plot = min(8, len(self.structure_names))
        selected_structures = self.structure_names[:n_structures_to_plot]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # ROC curves
        for i, struct_name in enumerate(selected_structures):
            y_true = presence_targets[:, i]
            y_scores = presence_predictions[:, i]

            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = metrics['presence_per_structure'][struct_name]['auc_roc']
                axes[0].plot(fpr, tpr, label=f'{struct_name[:8]} (AUC={auc:.3f})')

        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # PR curves
        for i, struct_name in enumerate(selected_structures):
            y_true = presence_targets[:, i]
            y_scores = presence_predictions[:, i]

            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                auc_pr = metrics['presence_per_structure'][struct_name]['auc_pr']
                axes[1].plot(recall, precision, label=f'{struct_name[:8]} (AUC={auc_pr:.3f})')

        baseline = presence_targets.mean(axis=0)
        for i, struct_name in enumerate(selected_structures):
            if i < len(baseline):
                axes[1].axhline(y=baseline[i], color=f'C{i}', linestyle='--', alpha=0.3)

        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_pr_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_detailed_results(self, metrics):
        """Save detailed results to files"""
        print("Saving detailed results...")

        # Save metrics as JSON
        with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create summary report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall statistics
        report_lines.append(f"Total test samples: {len(self.results['images'])}")
        report_lines.append(f"Number of anatomical structures: {len(self.structure_names)}")
        report_lines.append(f"Overall presence detection accuracy: {metrics['presence_overall_accuracy']:.4f}")
        report_lines.append(f"Overall mean Dice score: {metrics['segmentation_overall']['mean_dice']:.4f}")
        report_lines.append(f"Overall mean IoU score: {metrics['segmentation_overall']['mean_iou']:.4f}")
        report_lines.append("")

        # Per-structure presence detection performance
        report_lines.append("PRESENCE DETECTION PERFORMANCE BY STRUCTURE:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Structure':<15} {'Accuracy':<10} {'F1':<8} {'AUC-ROC':<8} {'Support':<8}")
        report_lines.append("-" * 60)

        for struct_name in self.structure_names:
            metrics_struct = metrics['presence_per_structure'][struct_name]
            report_lines.append(
                f"{struct_name[:14]:<15} "
                f"{metrics_struct['accuracy']:<10.4f} "
                f"{metrics_struct['f1_score']:<8.4f} "
                f"{metrics_struct['auc_roc']:<8.4f} "
                f"{metrics_struct['support_positive']:<8d}"
            )

        report_lines.append("")

        # Per-structure segmentation performance
        report_lines.append("SEGMENTATION PERFORMANCE BY STRUCTURE:")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Structure':<15} {'Mean Dice':<12} {'Mean IoU':<12} {'Samples':<8}")
        report_lines.append("-" * 50)

        for struct_name in self.structure_names:
            seg_metrics = metrics['segmentation_per_structure'][struct_name]
            report_lines.append(
                f"{struct_name[:14]:<15} "
                f"{seg_metrics['mean_dice']:<12.4f} "
                f"{seg_metrics['mean_iou']:<12.4f} "
                f"{seg_metrics['num_samples']:<8d}"
            )

        report_lines.append("")

        # Scenario-based performance
        report_lines.append("PERFORMANCE BY SCENARIO:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Scenario':<25} {'Accuracy':<10} {'Samples':<8}")
        report_lines.append("-" * 60)

        for scenario_name in sorted(metrics['scenario_based'].keys()):
            scenario_metrics = metrics['scenario_based'][scenario_name]
            report_lines.append(
                f"{scenario_name[:24]:<25} "
                f"{scenario_metrics['accuracy']:<10.4f} "
                f"{scenario_metrics['num_samples']:<8d}"
            )

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))

        # Save predictions for further analysis
        predictions_data = {
            'sample_indices': self.results['sample_indices'],
            'scenarios': self.results['scenarios'],
            'presence_predictions': self.results['presence_predictions'],
            'presence_targets': self.results['presence_targets'],
            'confidence_scores': self.results['confidence_scores']
        }

        # Convert numpy arrays to lists for JSON serialization
        for key, value in predictions_data.items():
            if isinstance(value, np.ndarray):
                predictions_data[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                predictions_data[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]

        with open(self.output_dir / 'predictions_data.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)

        print(f"  Detailed results saved to {self.output_dir}")


def load_model_from_checkpoint(checkpoint_path, structure_names, image_size, device):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint with weights_only=False for compatibility with older checkpoints
    # This is safe if you trust the source of your checkpoint files
    try:
        # Try loading with weights_only=True first (more secure)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"  Secure loading failed, trying compatibility mode...")
        print(f"  Error: {str(e)[:100]}...")
        # Fall back to weights_only=False for compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"  Loaded successfully with compatibility mode")

    # Create model
    model = SpatialAttentionMedicalSegmenter(
        in_channels=1,
        num_structures=len(structure_names),
        feature_channels=64,
        spatial_dims=image_size
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    return model


def create_attention_analysis_report(evaluator):
    """Create a detailed analysis of attention mechanisms"""
    print("Creating attention analysis report...")

    attention_maps = np.array(evaluator.results['attention_maps'])  # [N, C, D, H, W]
    position_priors = np.array(evaluator.results['position_priors'])  # [N, C, D, H, W]
    presence_targets = np.array(evaluator.results['presence_targets'])  # [N, C]

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ATTENTION MECHANISM ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Analyze attention distribution for each structure
    report_lines.append("ATTENTION DISTRIBUTION ANALYSIS:")
    report_lines.append("-" * 40)

    for i, struct_name in enumerate(evaluator.structure_names):
        # Calculate attention statistics for present structures only
        present_mask = presence_targets[:, i] == 1
        if present_mask.sum() == 0:
            continue

        present_attention = attention_maps[present_mask, i]  # [N_present, D, H, W]

        # Calculate statistics
        mean_attention = np.mean(present_attention)
        std_attention = np.std(present_attention)
        max_attention = np.max(present_attention)
        sparsity = (present_attention < 0.1).mean()  # Fraction of low attention values

        report_lines.append(f"{struct_name}:")
        report_lines.append(f"  Mean attention: {mean_attention:.4f}")
        report_lines.append(f"  Std attention: {std_attention:.4f}")
        report_lines.append(f"  Max attention: {max_attention:.4f}")
        report_lines.append(f"  Sparsity (<0.1): {sparsity:.4f}")
        report_lines.append("")

    # Analyze position priors
    report_lines.append("POSITION PRIORS ANALYSIS:")
    report_lines.append("-" * 30)

    for i, struct_name in enumerate(evaluator.structure_names):
        prior = position_priors[0, i]  # Take first sample (priors are the same)

        mean_prior = np.mean(prior)
        std_prior = np.std(prior)
        min_prior = np.min(prior)
        max_prior = np.max(prior)

        report_lines.append(f"{struct_name}:")
        report_lines.append(f"  Mean prior: {mean_prior:.4f}")
        report_lines.append(f"  Std prior: {std_prior:.4f}")
        report_lines.append(f"  Range: [{min_prior:.4f}, {max_prior:.4f}]")
        report_lines.append("")

    report_lines.append("=" * 60)

    # Save report
    with open(evaluator.output_dir / 'attention_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Comprehensive Medical Model Testing')

    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--dataset', '-d', type=str,
                        default='./synthetic_medical_dataset_hdf5',
                        help='Path to HDF5 dataset directory')
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output', '-o', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--max-samples', '-m', type=int, default=50,
                        help='Maximum number of samples to visualize individually')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for inference')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Check if weights file exists
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    # Check if dataset exists
    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

    print("=" * 80)
    print("COMPREHENSIVE MEDICAL MODEL TESTING")
    print("=" * 80)
    print(f"Weights file: {args.weights}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("")

    # Load dataset to get metadata
    print("Loading dataset metadata...")
    try:
        with open(Path(args.dataset) / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        structure_names = metadata['structure_names']
        image_size = metadata['image_size']

        print(f"  Found {len(structure_names)} anatomical structures")
        print(f"  Image size: {image_size}")
        print(f"  Structures: {structure_names}")

    except Exception as e:
        print(f"Error loading dataset metadata: {e}")
        return

    # Load model
    try:
        model = load_model_from_checkpoint(args.weights, structure_names, image_size, device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dataset and data loader
    print(f"\nCreating {args.split} dataset...")
    try:
        dataset = HDF5MedicalDataset(args.dataset, split=args.split)
        print(f"  Dataset loaded: {len(dataset)} samples")

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Create evaluator
    print(f"\nInitializing evaluator...")
    evaluator = ModelEvaluator(model, device, structure_names, args.output)

    # Run evaluation
    print(f"\nRunning evaluation on {len(dataset)} samples...")
    print("This may take a while depending on dataset size and hardware...")

    try:
        # Process all batches
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            evaluator.evaluate_batch(batch)

            # Optional: limit evaluation for testing
            # if batch_idx >= 10:  # Uncomment to limit evaluation
            #     break

        print(f"\nEvaluation completed! Processed {len(evaluator.results['images'])} samples")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Calculate metrics
    print("\nCalculating comprehensive metrics...")
    try:
        metrics = evaluator.calculate_metrics()
        print("  Metrics calculation completed!")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    # Create visualizations
    print("\nCreating visualizations...")
    try:
        evaluator.create_visualizations(metrics, max_samples=args.max_samples)
        print("  Visualizations completed!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return

    # Create attention analysis report
    print("\nCreating attention analysis...")
    try:
        create_attention_analysis_report(evaluator)
        print("  Attention analysis completed!")
    except Exception as e:
        print(f"Error creating attention analysis: {e}")
        return

    # Save detailed results
    print("\nSaving detailed results...")
    try:
        evaluator.save_detailed_results(metrics)
        print("  Results saved!")
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall presence detection accuracy: {metrics['presence_overall_accuracy']:.4f}")
    print(f"Overall mean Dice score: {metrics['segmentation_overall']['mean_dice']:.4f}")
    print(f"Overall mean IoU score: {metrics['segmentation_overall']['mean_iou']:.4f}")

    # Best and worst performing structures
    presence_f1_scores = {name: metrics['presence_per_structure'][name]['f1_score']
                          for name in structure_names}
    best_structure = max(presence_f1_scores, key=presence_f1_scores.get)
    worst_structure = min(presence_f1_scores, key=presence_f1_scores.get)

    print(f"\nBest performing structure: {best_structure} (F1: {presence_f1_scores[best_structure]:.4f})")
    print(f"Worst performing structure: {worst_structure} (F1: {presence_f1_scores[worst_structure]:.4f})")

    print(f"\nAll results saved to: {args.output}")
    print("Check the following files:")
    print("  - evaluation_report.txt: Detailed text report")
    print("  - attention_analysis_report.txt: Attention mechanism analysis")
    print("  - detailed_metrics.json: All metrics in JSON format")
    print("  - detailed_sample_*.png: Comprehensive sample analysis with attention maps")
    print("  - attention_maps_overview.png: Attention maps across structures")
    print("  - anatomical_priors.png: Learned anatomical position priors")
    print("  - segmentation_comparison_*.png: True vs predicted segmentations")
    print("  - sample_predictions_*.png: Individual sample visualizations")
    print("  - presence_detection_metrics.png: Presence detection performance")
    print("  - segmentation_metrics.png: Segmentation performance")
    print("  - scenario_analysis.png: Performance by scenario")
    print("  - confidence_analysis.png: Model confidence analysis")
    print("  - confusion_matrices.png: Confusion matrices for all structures")
    print("  - roc_pr_curves.png: ROC and Precision-Recall curves")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
