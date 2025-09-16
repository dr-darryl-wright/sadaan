import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import h5py
from tqdm import tqdm
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# Import your model and dataset classes
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sadaan import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss


class HDF5MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, split='test', indices=None, transform=None):
        self.hdf5_path = str(hdf5_path)
        self.split = split
        self.transform = transform
        self.indices = indices

        self._swmr_supported = getattr(h5py.get_config(), "swmr_support", False)
        self.use_swmr = False

        # Read metadata once
        with h5py.File(self.hdf5_path, "r") as f:
            if self.split in f and 'images' in f[self.split]:
                root = f[self.split]
            else:
                root = f
            self.n_samples = len(root['images'])
            self.structure_names = list(f.attrs.get('structure_names', [])) if 'structure_names' in f.attrs else None
            self.image_size = tuple(f.attrs.get('image_size')) if 'image_size' in f.attrs else None

        if self.indices is None:
            self.indices = list(range(self.n_samples))

        # Try SWMR mode if supported
        if self._swmr_supported:
            try:
                with h5py.File(self.hdf5_path, "r", swmr=True, libver="latest") as f:
                    if self.split in f and 'images' in f[self.split]:
                        _ = f[self.split]['images'][0]
                    else:
                        _ = f['images'][0]
                self.use_swmr = True
                print(f"[HDF5MedicalDataset] Using SWMR mode for {self.hdf5_path}")
            except (OSError, ValueError, IOError):
                self.use_swmr = False
                print(f"[HDF5MedicalDataset] SWMR not usable, falling back to normal read mode")
        else:
            print("[HDF5MedicalDataset] h5py not built with SWMR support; using normal read mode")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        open_kwargs = {"mode": "r"}
        if self.use_swmr:
            open_kwargs.update({"swmr": True, "libver": "latest"})

        with h5py.File(self.hdf5_path, **open_kwargs) as f:
            if self.split in f and 'images' in f[self.split]:
                root = f[self.split]
            else:
                root = f
            image = np.array(root['images'][real_idx], dtype=np.float32)
            masks = np.array(root['masks'][real_idx], dtype=np.float32)
            presence = np.array(root['presence_labels'][real_idx])
            scenario = root['scenarios'][real_idx]
            if isinstance(scenario, bytes):
                try:
                    scenario = scenario.decode("utf-8")
                except Exception:
                    scenario = str(scenario)

        image = torch.from_numpy(image).float()
        masks = torch.from_numpy(masks).float()
        presence = torch.from_numpy(presence).long()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "masks": masks,
            "presence_labels": presence,
            "scenario": scenario,
            "index": real_idx
        }
            'image': image,
            'masks': masks,
            'presence_labels': presence,
            'scenario': scenario,
            'index': real_idx
        }

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
            'raw_logits': [],
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

                # Store attention maps and other model internals
                attention_maps_np = outputs['attention_maps'][i].cpu().numpy()
                position_priors_np = outputs['position_priors'][i].cpu().numpy()
                feature_responses_np = outputs['feature_responses'][i].cpu().numpy()
                raw_logits_np = outputs['raw_logits'][i].cpu().numpy()

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
                self.results['raw_logits'].append(raw_logits_np)
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

        # ============ ATTENTION ANALYSIS METRICS ============
        print("  Computing attention analysis metrics...")
        attention_metrics = self.analyze_attention_quality()
        metrics['attention_analysis'] = attention_metrics

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

    def analyze_attention_quality(self):
        """Analyze quality of attention maps"""
        attention_metrics = {}

        for i, struct_name in enumerate(self.structure_names):
            attention_dice_scores = []
            attention_iou_scores = []
            attention_coverage_scores = []

            for sample_idx in range(len(self.results['attention_maps'])):
                if self.results['presence_targets'][sample_idx][i] == 1:
                    # Get attention map and true mask
                    attention_map = self.results['attention_maps'][sample_idx][i]
                    true_mask = self.results['segmentation_targets'][sample_idx][i]

                    # Binarize attention map
                    attention_binary = (attention_map > 0.5).astype(float)

                    # Calculate overlap metrics
                    att_dice = self.dice_coefficient(attention_binary, true_mask)
                    att_iou = self.iou_score(attention_binary, true_mask)

                    # Coverage: how much of the true structure is covered by attention
                    true_positive_voxels = (attention_binary * true_mask).sum()
                    total_true_voxels = true_mask.sum()
                    coverage = true_positive_voxels / (total_true_voxels + 1e-8)

                    attention_dice_scores.append(att_dice)
                    attention_iou_scores.append(att_iou)
                    attention_coverage_scores.append(coverage)

            attention_metrics[struct_name] = {
                'attention_dice_mean': np.mean(attention_dice_scores) if attention_dice_scores else 0.0,
                'attention_dice_std': np.std(attention_dice_scores) if attention_dice_scores else 0.0,
                'attention_iou_mean': np.mean(attention_iou_scores) if attention_iou_scores else 0.0,
                'attention_iou_std': np.std(attention_iou_scores) if attention_iou_scores else 0.0,
                'attention_coverage_mean': np.mean(attention_coverage_scores) if attention_coverage_scores else 0.0,
                'attention_coverage_std': np.std(attention_coverage_scores) if attention_coverage_scores else 0.0,
                'num_samples': len(attention_dice_scores)
            }

        return attention_metrics

    def create_visualizations(self, metrics, max_samples=50):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # ============ ENHANCED SAMPLE VISUALIZATIONS ============
        print("  Creating enhanced sample visualizations...")
        self.visualize_enhanced_samples(max_samples)

        # ============ ATTENTION MAPS ANALYSIS ============
        print("  Creating attention maps analysis...")
        self.visualize_attention_analysis(metrics)

        # ============ STRUCTURAL PRIORS VISUALIZATION ============
        print("  Creating structural priors visualization...")
        self.visualize_structural_priors()

        # ============ SEGMENTATION OVERLAYS ============
        print("  Creating segmentation overlays...")
        self.visualize_segmentation_overlays(max_samples // 2)

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

        # ============ FEATURE RESPONSE ANALYSIS ============
        print("  Creating feature response analysis...")
        self.visualize_feature_responses()

        print("  Visualizations completed!")

    def visualize_enhanced_samples(self, max_samples=50):
        """Enhanced sample visualization with attention and segmentation"""
        n_samples = min(max_samples, len(self.results['images']))
        samples_per_figure = 3
        n_figures = (n_samples + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(n_figures):
            start_idx = fig_idx * samples_per_figure
            end_idx = min(start_idx + samples_per_figure, n_samples)
            current_samples = end_idx - start_idx

            fig, axes = plt.subplots(current_samples, 8, figsize=(32, 4 * current_samples))
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
                axes[sample_idx, 0].set_title(f'Original\nScenario: {scenario[:20]}...')
                axes[sample_idx, 0].axis('off')

                # True segmentation overlay
                true_seg_overlay = img_slice.copy()
                for struct_idx in range(min(3, len(self.structure_names))):  # Show first 3 structures
                    if presence_true[struct_idx] == 1:
                        true_mask_slice = self.results['segmentation_targets'][idx][struct_idx][..., middle_slice]
                        true_mask_slice = np.rot90(true_mask_slice, -1)
                        true_seg_overlay[true_mask_slice > 0.5] = 150 + struct_idx * 30

                axes[sample_idx, 1].imshow(true_seg_overlay, cmap='viridis')
                axes[sample_idx, 1].set_title('True Segmentation\n(First 3 structures)')
                axes[sample_idx, 1].axis('off')

                # Predicted segmentation overlay
                pred_seg_overlay = img_slice.copy()
                for struct_idx in range(min(3, len(self.structure_names))):
                    if presence_pred[struct_idx] > 0.5:
                        pred_mask_slice = self.results['segmentation_predictions'][idx][struct_idx][..., middle_slice]
                        pred_mask_slice = np.rot90(pred_mask_slice, -1)
                        pred_seg_overlay[pred_mask_slice > 0.5] = 150 + struct_idx * 30

                axes[sample_idx, 2].imshow(pred_seg_overlay, cmap='viridis')
                axes[sample_idx, 2].set_title('Predicted Segmentation\n(First 3 structures)')
                axes[sample_idx, 2].axis('off')

                # Attention maps for first 3 structures
                for att_idx in range(min(3, len(self.structure_names))):
                    attention_slice = self.results['attention_maps'][idx][att_idx][..., middle_slice]
                    attention_slice = np.rot90(attention_slice, -1)

                    im = axes[sample_idx, 3 + att_idx].imshow(attention_slice, cmap='hot', vmin=0, vmax=1)
                    axes[sample_idx, 3 + att_idx].set_title(f'Attention: {self.structure_names[att_idx][:8]}')
                    axes[sample_idx, 3 + att_idx].axis('off')

                    # Add colorbar
                    divider = make_axes_locatable(axes[sample_idx, 3 + att_idx])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                # Position priors for first structure
                if len(self.results['position_priors']) > 0:
                    prior_slice = self.results['position_priors'][idx][0][..., middle_slice]
                    prior_slice = np.rot90(prior_slice, -1)

                    im = axes[sample_idx, 6].imshow(prior_slice, cmap='plasma', vmin=0, vmax=1)
                    axes[sample_idx, 6].set_title(f'Position Prior\n{self.structure_names[0][:8]}')
                    axes[sample_idx, 6].axis('off')

                    divider = make_axes_locatable(axes[sample_idx, 6])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                # Presence predictions summary
                presence_pred_binary = (presence_pred > 0.5).astype(int)
                correct_predictions = (presence_pred_binary == presence_true).astype(float)
                conf_scores = self.results['confidence_scores'][idx]

                y_pos = np.arange(min(8, len(self.structure_names)))
                colors = ['red' if c == 0 else 'green' for c in correct_predictions[:len(y_pos)]]

                bars = axes[sample_idx, 7].barh(y_pos, presence_pred[:len(y_pos)], color=colors, alpha=0.7)
                axes[sample_idx, 7].set_yticks(y_pos)
                axes[sample_idx, 7].set_yticklabels([name[:8] for name in self.structure_names[:len(y_pos)]],
                                                    fontsize=6)
                axes[sample_idx, 7].set_xlabel('Presence Probability')
                axes[sample_idx, 7].set_title('Presence Predictions\n(Green=Correct, Red=Wrong)')
                axes[sample_idx, 7].set_xlim(0, 1)
                axes[sample_idx, 7].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'enhanced_sample_predictions_{fig_idx + 1}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    def visualize_attention_analysis(self, metrics):
        """Visualize attention map quality analysis"""
        attention_metrics = metrics['attention_analysis']

        structures = list(attention_metrics.keys())
        attention_dice = [attention_metrics[s]['attention_dice_mean'] for s in structures]
        attention_iou = [attention_metrics[s]['attention_iou_mean'] for s in structures]
        attention_coverage = [attention_metrics[s]['attention_coverage_mean'] for s in structures]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Attention Dice scores
        x_pos = np.arange(len(structures))
        bars1 = axes[0, 0].bar(x_pos, attention_dice, color='lightblue', alpha=0.8)
        axes[0, 0].set_title('Attention Map Quality: Dice Overlap with True Masks')
        axes[0, 0].set_xlabel('Anatomical Structures')
        axes[0, 0].set_ylabel('Dice Score')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1)

        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Attention IoU scores
        bars2 = axes[0, 1].bar(x_pos, attention_iou, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Attention Map Quality: IoU Overlap with True Masks')
        axes[0, 1].set_xlabel('Anatomical Structures')
        axes[0, 1].set_ylabel('IoU Score')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1)

        # Attention coverage
        bars3 = axes[1, 0].bar(x_pos, attention_coverage, color='coral', alpha=0.8)
        axes[1, 0].set_title('Attention Coverage: % of True Structure Attended')
        axes[1, 0].set_xlabel('Anatomical Structures')
        axes[1, 0].set_ylabel('Coverage Ratio')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
        axes[1, 0].set_ylim(0, 1)

        # Scatter plot: attention quality vs segmentation quality
        seg_metrics = metrics['segmentation_per_structure']
        seg_dice = [seg_metrics[s]['mean_dice'] for s in structures]

        axes[1, 1].scatter(attention_dice, seg_dice, s=100, alpha=0.7, c='purple')
        for i, struct in enumerate(structures):
            axes[1, 1].annotate(struct[:6], (attention_dice[i], seg_dice[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Attention Quality (Dice)')
        axes[1, 1].set_ylabel('Segmentation Quality (Dice)')
        axes[1, 1].set_title('Attention Quality vs Segmentation Quality')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_structural_priors(self):
        """Visualize learned structural position priors"""
        if not self.results['position_priors']:
            print("    No position priors available, skipping visualization")
            return

        # Get average position priors across all samples
        all_priors = np.array(self.results['position_priors'])  # [N, num_structures, D, H, W]
        mean_priors = np.mean(all_priors, axis=0)  # [num_structures, D, H, W]

        n_structures = min(len(self.structure_names), mean_priors.shape[0])

        # Create visualization for middle slices (axial, sagittal, coronal)
        fig, axes = plt.subplots(n_structures, 4, figsize=(16, 4 * n_structures))
        if n_structures == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_structures):
            struct_name = self.structure_names[i]
            prior = mean_priors[i]  # [D, H, W]

            # Get middle slices
            mid_d, mid_h, mid_w = prior.shape[0] // 2, prior.shape[1] // 2, prior.shape[2] // 2

            # Axial slice (D dimension)
            axial_slice = np.rot90(prior[mid_d, :, :], -1)
            im1 = axes[i, 0].imshow(axial_slice, cmap='hot', vmin=0, vmax=prior.max())
            axes[i, 0].set_title(f'{struct_name}\nAxial (z={mid_d})')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # Sagittal slice (W dimension)
            sagittal_slice = np.rot90(prior[:, :, mid_w], -1)
            im2 = axes[i, 1].imshow(sagittal_slice, cmap='hot', vmin=0, vmax=prior.max())
            axes[i, 1].set_title(f'Sagittal (x={mid_w})')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Coronal slice (H dimension)
            coronal_slice = np.rot90(prior[:, mid_h, :], -1)
            im3 = axes[i, 2].imshow(coronal_slice, cmap='hot', vmin=0, vmax=prior.max())
            axes[i, 2].set_title(f'Coronal (y={mid_h})')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # 3D visualization as maximum intensity projection
            mip = np.max(prior, axis=0)  # Max along depth
            mip = np.rot90(mip, -1)
            im4 = axes[i, 3].imshow(mip, cmap='hot', vmin=0, vmax=mip.max())
            axes[i, 3].set_title('Max Intensity Projection')
            axes[i, 3].axis('off')
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)

        plt.suptitle('Learned Structural Position Priors', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'structural_position_priors.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_segmentation_overlays(self, max_samples=25):
        """Create detailed segmentation overlays with attention maps"""
        n_samples = min(max_samples, len(self.results['images']))

        for sample_idx in range(n_samples):
            # Create a detailed visualization for each sample
            fig, axes = plt.subplots(3, len(self.structure_names),
                                     figsize=(4 * len(self.structure_names), 12))

            image = self.results['images'][sample_idx][0]
            scenario = self.results['scenarios'][sample_idx]
            presence_true = self.results['presence_targets'][sample_idx]
            presence_pred = self.results['presence_predictions'][sample_idx]

            # Get middle slice
            middle_slice = image.shape[-1] // 2
            img_slice = np.rot90(image[..., middle_slice], -1)

            for struct_idx, struct_name in enumerate(self.structure_names):
                # Row 1: True mask overlay
                true_overlay = img_slice.copy()
                if presence_true[struct_idx] == 1:
                    true_mask = self.results['segmentation_targets'][sample_idx][struct_idx]
                    true_mask_slice = np.rot90(true_mask[..., middle_slice], -1)
                    true_overlay[true_mask_slice > 0.5] = 255

                axes[0, struct_idx].imshow(true_overlay, cmap='gray')
                axes[0, struct_idx].set_title(f'{struct_name}\nTrue (Present: {presence_true[struct_idx]})')
                axes[0, struct_idx].axis('off')

                # Row 2: Predicted mask overlay
                pred_overlay = img_slice.copy()
                pred_mask = self.results['segmentation_predictions'][sample_idx][struct_idx]
                pred_mask_slice = np.rot90(pred_mask[..., middle_slice], -1)
                pred_overlay[pred_mask_slice > 0.5] = 255

                axes[1, struct_idx].imshow(pred_overlay, cmap='gray')
                axes[1, struct_idx].set_title(f'Predicted (Prob: {presence_pred[struct_idx]:.3f})')
                axes[1, struct_idx].axis('off')

                # Row 3: Attention map
                attention_map = self.results['attention_maps'][sample_idx][struct_idx]
                attention_slice = np.rot90(attention_map[..., middle_slice], -1)

                im = axes[2, struct_idx].imshow(attention_slice, cmap='hot', vmin=0, vmax=1)
                axes[2, struct_idx].set_title('Attention Map')
                axes[2, struct_idx].axis('off')

                # Add colorbar for attention
                divider = make_axes_locatable(axes[2, struct_idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            plt.suptitle(f'Sample {sample_idx}: {scenario}', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'detailed_segmentation_sample_{sample_idx:03d}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    def visualize_feature_responses(self):
        """Visualize feature response patterns"""
        if not self.results['feature_responses']:
            print("    No feature responses available, skipping visualization")
            return

        # Analyze feature response statistics
        all_responses = np.array(self.results['feature_responses'])  # [N, num_structures, D, H, W]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Mean feature response by structure
        mean_responses = np.mean(all_responses, axis=(0, 2, 3, 4))  # [num_structures]

        axes[0, 0].bar(range(len(self.structure_names)), mean_responses, color='skyblue')
        axes[0, 0].set_title('Mean Feature Response by Structure')
        axes[0, 0].set_xlabel('Anatomical Structures')
        axes[0, 0].set_ylabel('Mean Response')
        axes[0, 0].set_xticks(range(len(self.structure_names)))
        axes[0, 0].set_xticklabels([s[:8] for s in self.structure_names], rotation=45)

        # 2. Feature response variability
        std_responses = np.std(all_responses, axis=(0, 2, 3, 4))

        axes[0, 1].bar(range(len(self.structure_names)), std_responses, color='lightcoral')
        axes[0, 1].set_title('Feature Response Variability')
        axes[0, 1].set_xlabel('Anatomical Structures')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_xticks(range(len(self.structure_names)))
        axes[0, 1].set_xticklabels([s[:8] for s in self.structure_names], rotation=45)

        # 3. Response correlation between structures
        response_correlations = np.corrcoef(mean_responses)
        im = axes[0, 2].imshow(response_correlations, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 2].set_title('Inter-Structure Response Correlations')
        axes[0, 2].set_xticks(range(len(self.structure_names)))
        axes[0, 2].set_yticks(range(len(self.structure_names)))
        axes[0, 2].set_xticklabels([s[:6] for s in self.structure_names], rotation=45)
        axes[0, 2].set_yticklabels([s[:6] for s in self.structure_names])
        plt.colorbar(im, ax=axes[0, 2])

        # 4. Feature response vs presence accuracy
        presence_metrics = {}
        for i, struct_name in enumerate(self.structure_names):
            presence_true = np.array([sample[i] for sample in self.results['presence_targets']])
            presence_pred = np.array([sample[i] for sample in self.results['presence_predictions']])
            accuracy = np.mean((presence_pred > 0.5) == presence_true)
            presence_metrics[struct_name] = accuracy

        accuracies = [presence_metrics[s] for s in self.structure_names]

        axes[1, 0].scatter(mean_responses, accuracies, s=100, alpha=0.7, c='purple')
        for i, struct in enumerate(self.structure_names):
            axes[1, 0].annotate(struct[:6], (mean_responses[i], accuracies[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Mean Feature Response')
        axes[1, 0].set_ylabel('Presence Detection Accuracy')
        axes[1, 0].set_title('Feature Response vs Detection Accuracy')

        # 5. Response distribution across samples
        sample_means = np.mean(all_responses, axis=(1, 2, 3, 4))  # Mean across all structures and spatial dims

        axes[1, 1].hist(sample_means, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_xlabel('Mean Feature Response')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title('Distribution of Sample-wise Feature Responses')

        # 6. Feature response heatmap for first sample
        if len(all_responses) > 0:
            sample_response = all_responses[0]  # [num_structures, D, H, W]
            middle_slice = sample_response.shape[-1] // 2

            # Create a grid showing responses for all structures
            response_grid = np.zeros((len(self.structure_names) * sample_response.shape[1],
                                      sample_response.shape[2]))

            for i in range(len(self.structure_names)):
                start_row = i * sample_response.shape[1]
                end_row = (i + 1) * sample_response.shape[1]
                response_slice = sample_response[i, :, :, middle_slice]
                response_grid[start_row:end_row, :] = response_slice

            im = axes[1, 2].imshow(response_grid, cmap='viridis', aspect='auto')
            axes[1, 2].set_title('Feature Response Grid (Sample 1)')
            axes[1, 2].set_ylabel('Structure × Depth')
            axes[1, 2].set_xlabel('Width')

            # Add structure labels
            structure_positions = [(i + 0.5) * sample_response.shape[1]
                                   for i in range(len(self.structure_names))]
            axes[1, 2].set_yticks(structure_positions)
            axes[1, 2].set_yticklabels([s[:8] for s in self.structure_names])

            plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_response_analysis.png', dpi=150, bbox_inches='tight')
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
        axes[0, 0].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
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
        axes[1, 0].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
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
        axes[1, 1].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')

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
        axes[0].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
        axes[0].set_ylim(0, 1)

        # IoU scores with error bars
        bars2 = axes[1].bar(x_pos, iou_means, yerr=iou_stds, capsize=5, color='lightgreen', alpha=0.8)
        axes[1].set_title('IoU Scores by Structure')
        axes[1].set_xlabel('Anatomical Structures')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')
        axes[1].set_ylim(0, 1)

        # Number of samples used for evaluation
        bars3 = axes[2].bar(x_pos, num_samples, color='orange', alpha=0.8)
        axes[2].set_title('Number of Samples with Present Structure')
        axes[2].set_xlabel('Anatomical Structures')
        axes[2].set_ylabel('Number of Samples')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([s[:8] for s in structures], rotation=45, ha='right')

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

        # Attention analysis summary
        if 'attention_analysis' in metrics:
            report_lines.append("ATTENTION ANALYSIS SUMMARY:")
            report_lines.append("-" * 40)
            attention_metrics = metrics['attention_analysis']
            avg_attention_dice = np.mean([attention_metrics[s]['attention_dice_mean']
                                          for s in self.structure_names])
            avg_attention_coverage = np.mean([attention_metrics[s]['attention_coverage_mean']
                                              for s in self.structure_names])
            report_lines.append(f"Average attention-mask overlap (Dice): {avg_attention_dice:.4f}")
            report_lines.append(f"Average attention coverage: {avg_attention_coverage:.4f}")
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

        # Attention analysis per structure
        if 'attention_analysis' in metrics:
            report_lines.append("ATTENTION ANALYSIS BY STRUCTURE:")
            report_lines.append("-" * 60)
            report_lines.append(f"{'Structure':<15} {'Att Dice':<10} {'Att IoU':<10} {'Coverage':<10}")
            report_lines.append("-" * 60)

            for struct_name in self.structure_names:
                att_metrics = metrics['attention_analysis'][struct_name]
                report_lines.append(
                    f"{struct_name[:14]:<15} "
                    f"{att_metrics['attention_dice_mean']:<10.4f} "
                    f"{att_metrics['attention_iou_mean']:<10.4f} "
                    f"{att_metrics['attention_coverage_mean']:<10.4f}"
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


def create_additional_tests(evaluator, metrics):
    """Create additional specialized tests and analyses"""
    print("Running additional specialized tests...")

    # Test 1: Model behavior on edge cases
    print("  Testing edge cases...")
    edge_case_results = analyze_edge_cases(evaluator)

    # Test 2: Attention consistency analysis
    print("  Analyzing attention consistency...")
    attention_consistency = analyze_attention_consistency(evaluator)

    # Test 3: Multi-structure interaction analysis
    print("  Analyzing multi-structure interactions...")
    interaction_analysis = analyze_structure_interactions(evaluator)

    # Test 4: Model uncertainty analysis
    print("  Analyzing model uncertainty...")
    uncertainty_analysis = analyze_model_uncertainty(evaluator)

    additional_results = {
        'edge_cases': edge_case_results,
        'attention_consistency': attention_consistency,
        'structure_interactions': interaction_analysis,
        'uncertainty_analysis': uncertainty_analysis
    }

    # Save additional results
    with open(evaluator.output_dir / 'additional_tests.json', 'w') as f:
        json.dump(additional_results, f, indent=2)

    return additional_results


def analyze_edge_cases(evaluator):
    """Analyze model performance on edge cases"""
    edge_cases = {
        'high_confidence_errors': [],
        'low_confidence_correct': [],
        'attention_segmentation_mismatch': [],
        'absent_with_high_attention': []
    }

    for i, (conf_scores, presence_targets, presence_preds, attention_maps) in enumerate(
            zip(evaluator.results['confidence_scores'],
                evaluator.results['presence_targets'],
                evaluator.results['presence_predictions'],
                evaluator.results['attention_maps'])):

        presence_pred_binary = (np.array(presence_preds) > 0.5).astype(int)

        for j, struct_name in enumerate(evaluator.structure_names):
            conf = conf_scores[j]
            true_label = presence_targets[j]
            pred_label = presence_pred_binary[j]
            attention_mean = np.mean(attention_maps[j])

            # High confidence errors
            if conf > 0.8 and true_label != pred_label:
                edge_cases['high_confidence_errors'].append({
                    'sample_idx': i,
                    'structure': struct_name,
                    'confidence': float(conf),
                    'true_label': int(true_label),
                    'pred_label': int(pred_label)
                })

            # Low confidence correct predictions
            if conf < 0.3 and true_label == pred_label:
                edge_cases['low_confidence_correct'].append({
                    'sample_idx': i,
                    'structure': struct_name,
                    'confidence': float(conf),
                    'true_label': int(true_label),
                    'pred_label': int(pred_label)
                })

            # Absent structures with high attention
            if true_label == 0 and attention_mean > 0.5:
                edge_cases['absent_with_high_attention'].append({
                    'sample_idx': i,
                    'structure': struct_name,
                    'attention_mean': float(attention_mean),
                    'true_label': int(true_label)
                })

    return edge_cases


def analyze_attention_consistency(evaluator):
    """Analyze consistency of attention maps across similar samples"""
    # Group samples by scenario
    scenario_groups = defaultdict(list)
    for i, scenario in enumerate(evaluator.results['scenarios']):
        scenario_groups[scenario].append(i)

    consistency_metrics = {}

    for scenario, sample_indices in scenario_groups.items():
        if len(sample_indices) < 2:
            continue

        scenario_consistency = {}

        for struct_idx, struct_name in enumerate(evaluator.structure_names):
            attention_maps = []
            for sample_idx in sample_indices:
                if evaluator.results['presence_targets'][sample_idx][struct_idx] == 1:
                    attention_maps.append(evaluator.results['attention_maps'][sample_idx][struct_idx])

            if len(attention_maps) >= 2:
                # Calculate pairwise correlations
                correlations = []
                for i in range(len(attention_maps)):
                    for j in range(i + 1, len(attention_maps)):
                        map1_flat = attention_maps[i].flatten()
                        map2_flat = attention_maps[j].flatten()
                        corr = np.corrcoef(map1_flat, map2_flat)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    scenario_consistency[struct_name] = {
                        'mean_correlation': float(np.mean(correlations)),
                        'std_correlation': float(np.std(correlations)),
                        'num_comparisons': len(correlations)
                    }

        if scenario_consistency:
            consistency_metrics[scenario] = scenario_consistency

    return consistency_metrics


def analyze_structure_interactions(evaluator):
    """Analyze how the presence of one structure affects detection of others"""
    interaction_matrix = np.zeros((len(evaluator.structure_names), len(evaluator.structure_names)))

    presence_targets = np.array(evaluator.results['presence_targets'])
    presence_preds = np.array(evaluator.results['presence_predictions'])

    for i in range(len(evaluator.structure_names)):
        for j in range(len(evaluator.structure_names)):
            if i != j:
                # When structure i is present, what's the accuracy for structure j?
                mask_i_present = presence_targets[:, i] == 1
                if mask_i_present.sum() > 0:
                    j_accuracy_when_i_present = np.mean(
                        (presence_preds[mask_i_present, j] > 0.5) == presence_targets[mask_i_present, j]
                    )
                    interaction_matrix[i, j] = j_accuracy_when_i_present

    # Convert to dictionary format
    interactions = {}
    for i, struct_i in enumerate(evaluator.structure_names):
        interactions[struct_i] = {}
        for j, struct_j in enumerate(evaluator.structure_names):
            if i != j:
                interactions[struct_i][struct_j] = float(interaction_matrix[i, j])

    return interactions


def analyze_model_uncertainty(evaluator):
    """Analyze model uncertainty patterns"""
    uncertainty_metrics = {
        'entropy_by_structure': {},
        'uncertainty_vs_accuracy': {},
        'high_uncertainty_cases': []
    }

    presence_preds = np.array(evaluator.results['presence_predictions'])
    presence_targets = np.array(evaluator.results['presence_targets'])

    for i, struct_name in enumerate(evaluator.structure_names):
        probs = presence_preds[:, i]
        targets = presence_targets[:, i]

        # Calculate entropy (uncertainty)
        entropies = -probs * np.log(probs + 1e-8) - (1 - probs) * np.log(1 - probs + 1e-8)

        # Accuracy for each prediction
        accuracies = ((probs > 0.5).astype(int) == targets).astype(float)

        uncertainty_metrics['entropy_by_structure'][struct_name] = {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'entropy_accuracy_correlation': float(np.corrcoef(entropies, accuracies)[0, 1])
        }

        # Find high uncertainty cases
        high_uncertainty_threshold = np.percentile(entropies, 90)
        high_unc_indices = np.where(entropies > high_uncertainty_threshold)[0]

        for idx in high_unc_indices:
            uncertainty_metrics['high_uncertainty_cases'].append({
                'sample_idx': int(idx),
                'structure': struct_name,
                'entropy': float(entropies[idx]),
                'probability': float(probs[idx]),
                'true_label': int(targets[idx]),
                'correct': bool(accuracies[idx])
            })

    return uncertainty_metrics


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Enhanced Medical Model Testing with Advanced Visualizations')

    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--dataset', '-d', type=str,
                        default='./synthetic_medical_dataset_hdf5',
                        help='Path to HDF5 dataset directory')
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output', '-o', type=str, default='./enhanced_test_results',
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
    parser.add_argument('--detailed-segmentation', action='store_true',
                        help='Create detailed segmentation visualizations for all samples')
    parser.add_argument('--skip-additional-tests', action='store_true',
                        help='Skip additional specialized tests (faster execution)')

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
    print("ENHANCED MEDICAL MODEL TESTING WITH ADVANCED VISUALIZATIONS")
    print("=" * 80)
    print(f"Weights file: {args.weights}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Detailed segmentation: {args.detailed_segmentation}")
    print(f"Additional tests: {not args.skip_additional_tests}")
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
    print(f"\nInitializing enhanced evaluator...")
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
    print("\nCreating enhanced visualizations...")
    try:
        max_viz_samples = args.max_samples if not args.detailed_segmentation else len(evaluator.results['images'])
        evaluator.create_visualizations(metrics, max_samples=max_viz_samples)
        print("  Visualizations completed!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return

    # Run additional specialized tests
    if not args.skip_additional_tests:
        try:
            additional_results = create_additional_tests(evaluator, metrics)
            print("  Additional tests completed!")
        except Exception as e:
            print(f"Error in additional tests: {e}")
            additional_results = None
    else:
        additional_results = None

    # Save detailed results
    print("\nSaving detailed results...")
    try:
        evaluator.save_detailed_results(metrics)
        print("  Results saved!")
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    # Print enhanced summary
    print("\n" + "=" * 80)
    print("ENHANCED EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall presence detection accuracy: {metrics['presence_overall_accuracy']:.4f}")
    print(f"Overall mean Dice score: {metrics['segmentation_overall']['mean_dice']:.4f}")
    print(f"Overall mean IoU score: {metrics['segmentation_overall']['mean_iou']:.4f}")

    # Attention analysis summary
    if 'attention_analysis' in metrics:
        attention_metrics = metrics['attention_analysis']
        avg_attention_dice = np.mean([attention_metrics[s]['attention_dice_mean']
                                      for s in structure_names])
        avg_attention_coverage = np.mean([attention_metrics[s]['attention_coverage_mean']
                                          for s in structure_names])
        print(f"Average attention quality (Dice): {avg_attention_dice:.4f}")
        print(f"Average attention coverage: {avg_attention_coverage:.4f}")

    # Best and worst performing structures
    presence_f1_scores = {name: metrics['presence_per_structure'][name]['f1_score']
                          for name in structure_names}
    best_structure = max(presence_f1_scores, key=presence_f1_scores.get)
    worst_structure = min(presence_f1_scores, key=presence_f1_scores.get)

    print(f"\nBest performing structure: {best_structure} (F1: {presence_f1_scores[best_structure]:.4f})")
    print(f"Worst performing structure: {worst_structure} (F1: {presence_f1_scores[worst_structure]:.4f})")

    # Additional test results summary
    if additional_results:
        print(f"\nAdditional Tests Summary:")
        edge_cases = additional_results['edge_cases']
        print(f"  High confidence errors: {len(edge_cases['high_confidence_errors'])}")
        print(f"  Low confidence correct predictions: {len(edge_cases['low_confidence_correct'])}")
        print(f"  Absent structures with high attention: {len(edge_cases['absent_with_high_attention'])}")

        if additional_results['uncertainty_analysis']:
            avg_entropy = np.mean([metrics['mean_entropy'] for metrics in
                                   additional_results['uncertainty_analysis']['entropy_by_structure'].values()])
            print(f"  Average prediction entropy: {avg_entropy:.4f}")

    # Print model insights
    print(f"\nModel Insights:")
    if 'attention_analysis' in metrics:
        # Find structures with best/worst attention quality
        attention_dice_scores = {name: metrics['attention_analysis'][name]['attention_dice_mean']
                                 for name in structure_names if metrics['attention_analysis'][name]['num_samples'] > 0}
        if attention_dice_scores:
            best_attention = max(attention_dice_scores, key=attention_dice_scores.get)
            worst_attention = min(attention_dice_scores, key=attention_dice_scores.get)
            print(f"  Best attention quality: {best_attention} (Dice: {attention_dice_scores[best_attention]:.4f})")
            print(f"  Worst attention quality: {worst_attention} (Dice: {attention_dice_scores[worst_attention]:.4f})")

    # Scenario complexity analysis
    scenario_metrics = metrics['scenario_based']
    complexity_accuracy = [
        (scenario_metrics[scenario]['avg_structures_present'], scenario_metrics[scenario]['accuracy'])
        for scenario in scenario_metrics.keys()]
    if complexity_accuracy:
        avg_complexity = np.mean([x[0] for x in complexity_accuracy])
        if len(complexity_accuracy) > 1:
            complexity_acc_corr = np.corrcoef([x[0] for x in complexity_accuracy],
                                              [x[1] for x in complexity_accuracy])[0, 1]
        else:
            complexity_acc_corr = 0.0
        print(f"  Average scenario complexity: {avg_complexity:.2f} structures")
        print(f"  Complexity vs accuracy correlation: {complexity_acc_corr:.3f}")

    print(f"\nAll results saved to: {args.output}")
    print("Generated files:")
    print("=" * 50)
    print("Core Analysis:")
    print("  - evaluation_report.txt: Comprehensive text report")
    print("  - detailed_metrics.json: All metrics in JSON format")
    print("  - predictions_data.json: Raw predictions for further analysis")
    print("")
    print("Visualizations:")
    print("  - enhanced_sample_predictions_*.png: Enhanced sample visualizations")
    print("  - attention_analysis.png: Attention map quality analysis")
    print("  - structural_position_priors.png: Learned anatomical position priors")
    if args.detailed_segmentation:
        print("  - detailed_segmentation_sample_*.png: Individual segmentation overlays")
    print("  - feature_response_analysis.png: Feature response patterns")
    print("  - presence_detection_metrics.png: Presence detection performance")
    print("  - segmentation_metrics.png: Segmentation performance")
    print("  - scenario_analysis.png: Performance by scenario")
    print("  - confidence_analysis.png: Model confidence and calibration")
    print("  - confusion_matrices.png: Confusion matrices for all structures")
    print("  - roc_pr_curves.png: ROC and Precision-Recall curves")
    print("")
    if not args.skip_additional_tests:
        print("Advanced Analysis:")
        print("  - additional_tests.json: Edge cases and specialized analysis")
        print("")

    print("Recommended Next Steps:")
    print("-" * 30)

    # Provide recommendations based on results
    overall_performance = metrics['presence_overall_accuracy']
    seg_performance = metrics['segmentation_overall']['mean_dice']

    if overall_performance < 0.7:
        print("• Overall presence detection accuracy is low (<70%)")
        print("  - Review training data balance and augmentation")
        print("  - Consider adjusting loss function weights")
        print("  - Check attention_analysis.png for attention quality issues")

    if seg_performance < 0.5:
        print("• Segmentation performance is low (Dice <50%)")
        print("  - Review structural_position_priors.png for anatomical learning")
        print("  - Check feature_response_analysis.png for feature quality")
        print("  - Consider multi-scale feature extraction")

    if additional_results and len(additional_results['edge_cases']['high_confidence_errors']) > 10:
        print("• Model shows overconfidence in many wrong predictions")
        print("  - Implement better uncertainty quantification")
        print("  - Review confidence_analysis.png for calibration issues")
        print("  - Consider ensemble methods or temperature scaling")

    if 'attention_analysis' in metrics:
        attention_samples = [metrics['attention_analysis'][s]['num_samples']
                             for s in structure_names if metrics['attention_analysis'][s]['num_samples'] > 0]
        if attention_samples:
            avg_att_dice = np.mean([metrics['attention_analysis'][s]['attention_dice_mean']
                                    for s in structure_names if metrics['attention_analysis'][s]['num_samples'] > 0])
            if avg_att_dice < 0.3:
                print("• Attention maps poorly aligned with true structures")
                print("  - Review attention supervision loss weights")
                print("  - Check enhanced_sample_predictions_*.png for attention patterns")
                print("  - Consider stronger anatomical constraints")

    print("• Review detailed_metrics.json for structure-specific insights")
    print("• Check scenario_analysis.png for performance patterns across different cases")

    print("\n" + "=" * 80)
    print("Testing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
