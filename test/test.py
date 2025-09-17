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
import gc

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


class MemoryEfficientEvaluator:
    """Memory-efficient model evaluation with streaming processing"""

    def __init__(self, model, device, structure_names, output_dir='./test_results'):
        self.model = model
        self.device = device
        self.structure_names = structure_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lightweight accumulators instead of storing all data
        self.n_samples_processed = 0
        self.n_structures = len(structure_names)

        # Streaming metrics accumulators
        self.presence_confusion_matrices = np.zeros((self.n_structures, 2, 2), dtype=np.int64)
        self.presence_scores_sum = np.zeros(self.n_structures, dtype=np.float64)
        self.presence_true_labels = []  # Store only for final ROC/PR curves
        self.presence_pred_scores = []  # Store only for final ROC/PR curves

        # Segmentation metrics accumulators
        self.dice_scores = [[] for _ in range(self.n_structures)]
        self.iou_scores = [[] for _ in range(self.n_structures)]

        # Scenario tracking
        self.scenario_stats = defaultdict(
            lambda: {'correct': 0, 'total': 0, 'structures_present': [], 'structures_predicted': []})

        # For visualization - store only selected samples
        self.visualization_samples = []
        self.max_vis_samples = 20  # Limit visualization samples

        # Attention analysis data (lightweight)
        self.attention_stats = defaultdict(lambda: {'mean': 0, 'var': 0, 'n': 0})
        self.prior_stats = defaultdict(lambda: {'mean': 0, 'min': float('inf'), 'max': float('-inf')})

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

    def update_running_stats(self, values, stats_dict, key):
        """Update running statistics for streaming calculation"""
        values = np.array(values)
        n = stats_dict[key]['n']
        old_mean = stats_dict[key]['mean']
        old_var = stats_dict[key]['var']

        # Update count
        new_n = n + len(values)

        # Update mean
        new_mean = (old_mean * n + values.sum()) / new_n

        # Update variance (Welford's online algorithm)
        if n > 0:
            delta = values.mean() - old_mean
            new_var = (old_var * n + ((values - old_mean) * (values - new_mean)).sum()) / new_n
        else:
            new_var = values.var() if len(values) > 1 else 0

        stats_dict[key]['mean'] = new_mean
        stats_dict[key]['var'] = new_var
        stats_dict[key]['n'] = new_n

    def evaluate_sample(self, sample):
        """Evaluate a single sample - memory efficient"""
        self.model.eval()

        with torch.no_grad():
            # Process single sample
            image = sample['image'].unsqueeze(0).to(self.device)  # Add batch dim
            true_masks = sample['masks'].unsqueeze(0).to(self.device)
            true_presence = sample['presence_labels'].unsqueeze(0).to(self.device)
            scenario = sample['scenario']
            index = sample['index']

            # Forward pass
            outputs = self.model(image)

            # Move to CPU and convert to numpy immediately
            pred_masks_np = outputs['segmentation_probs'][0].cpu().numpy()
            pred_presence_np = outputs['presence_probs'][0].cpu().numpy()
            true_masks_np = true_masks[0].cpu().numpy()
            true_presence_np = true_presence[0].cpu().numpy()

            # Store attention stats (lightweight)
            if self.n_samples_processed < 5:  # Only for first few samples
                attention_maps_np = outputs['attention_maps'][0].cpu().numpy()
                position_priors_np = outputs['position_priors'][0].cpu().numpy()

                for i, struct_name in enumerate(self.structure_names):
                    # Update attention statistics
                    attention_values = attention_maps_np[i].flatten()
                    self.update_running_stats(attention_values, self.attention_stats, struct_name)

                    # Update prior statistics (these should be the same across samples)
                    prior_values = position_priors_np[i]
                    self.prior_stats[struct_name]['mean'] = prior_values.mean()
                    self.prior_stats[struct_name]['min'] = prior_values.min()
                    self.prior_stats[struct_name]['max'] = prior_values.max()

            # Clear GPU memory
            del outputs, image, true_masks, true_presence
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None

            # Process presence detection
            pred_presence_binary = (pred_presence_np > 0.5).astype(int)

            # Update confusion matrices
            for i in range(self.n_structures):
                true_label = int(true_presence_np[i])
                pred_label = int(pred_presence_binary[i])
                self.presence_confusion_matrices[i, true_label, pred_label] += 1

            # Store for final ROC/PR curves (keep lightweight)
            if self.n_samples_processed % 10 == 0 or self.n_samples_processed < 100:  # Sample subset
                self.presence_true_labels.append(true_presence_np.copy())
                self.presence_pred_scores.append(pred_presence_np.copy())

            # Process segmentation (only for present structures)
            pred_masks_binary = (pred_masks_np > 0.5).astype(float)

            for i in range(self.n_structures):
                if true_presence_np[i] == 1:  # Only calculate for present structures
                    dice = self.dice_coefficient(pred_masks_binary[i], true_masks_np[i])
                    iou = self.iou_score(pred_masks_binary[i], true_masks_np[i])
                    self.dice_scores[i].append(dice)
                    self.iou_scores[i].append(iou)

            # Update scenario statistics
            sample_correct = (pred_presence_binary == true_presence_np).all()
            self.scenario_stats[scenario]['correct'] += int(sample_correct)
            self.scenario_stats[scenario]['total'] += 1
            self.scenario_stats[scenario]['structures_present'].append(true_presence_np.sum())
            self.scenario_stats[scenario]['structures_predicted'].append(pred_presence_binary.sum())

            # Store selected samples for visualization
            if len(self.visualization_samples) < self.max_vis_samples:
                if (self.n_samples_processed % max(1, self.n_samples_processed // self.max_vis_samples) == 0 or
                        self.n_samples_processed < self.max_vis_samples):
                    # Store minimal visualization data
                    vis_sample = {
                        'index': index,
                        'scenario': scenario,
                        'image_slice': sample['image'][0, :, :, sample['image'].shape[-1] // 2].numpy(),
                        # Middle slice only
                        'true_presence': true_presence_np.copy(),
                        'pred_presence': pred_presence_np.copy(),
                        'confidence': 1.0 - (-pred_presence_np * np.log(pred_presence_np + 1e-8) -
                                             (1 - pred_presence_np) * np.log(1 - pred_presence_np + 1e-8))
                    }
                    self.visualization_samples.append(vis_sample)

            self.n_samples_processed += 1

            # Periodic cleanup
            if self.n_samples_processed % 50 == 0:
                gc.collect()

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics from accumulated data"""
        print("Calculating evaluation metrics from accumulated data...")

        metrics = {}

        # ============ PRESENCE DETECTION METRICS ============
        print("  Computing presence detection metrics...")

        # Calculate metrics from confusion matrices
        presence_metrics = {}
        overall_correct = 0
        overall_total = 0

        for i, struct_name in enumerate(self.structure_names):
            cm = self.presence_confusion_matrices[i]
            tp, tn, fp, fn = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]

            total = tp + tn + fp + fn
            overall_correct += tp + tn
            overall_total += total

            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            presence_metrics[struct_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'support_positive': int(tp + fn),
                'support_negative': int(tn + fp)
            }

        metrics['presence_overall_accuracy'] = float(overall_correct / overall_total) if overall_total > 0 else 0.0
        metrics['presence_per_structure'] = presence_metrics

        # Calculate AUC scores from stored subset
        if self.presence_true_labels and self.presence_pred_scores:
            true_labels = np.array(self.presence_true_labels)
            pred_scores = np.array(self.presence_pred_scores)

            for i, struct_name in enumerate(self.structure_names):
                y_true = true_labels[:, i]
                y_scores = pred_scores[:, i]

                try:
                    if len(np.unique(y_true)) > 1:
                        auc_roc = roc_auc_score(y_true, y_scores)
                        auc_pr = average_precision_score(y_true, y_scores)
                    else:
                        auc_roc = 0.5
                        auc_pr = y_true.mean()
                except:
                    auc_roc = 0.5
                    auc_pr = y_true.mean() if len(y_true) > 0 else 0.5

                metrics['presence_per_structure'][struct_name]['auc_roc'] = float(auc_roc)
                metrics['presence_per_structure'][struct_name]['auc_pr'] = float(auc_pr)

        # ============ SEGMENTATION METRICS ============
        print("  Computing segmentation metrics...")
        segmentation_metrics = {}
        all_dice = []
        all_iou = []

        for i, struct_name in enumerate(self.structure_names):
            dice_list = self.dice_scores[i]
            iou_list = self.iou_scores[i]

            if dice_list:
                segmentation_metrics[struct_name] = {
                    'mean_dice': float(np.mean(dice_list)),
                    'std_dice': float(np.std(dice_list)),
                    'mean_iou': float(np.mean(iou_list)),
                    'std_iou': float(np.std(iou_list)),
                    'num_samples': len(dice_list)
                }
                all_dice.append(np.mean(dice_list))
                all_iou.append(np.mean(iou_list))
            else:
                segmentation_metrics[struct_name] = {
                    'mean_dice': 0.0,
                    'std_dice': 0.0,
                    'mean_iou': 0.0,
                    'std_iou': 0.0,
                    'num_samples': 0
                }

        metrics['segmentation_per_structure'] = segmentation_metrics
        metrics['segmentation_overall'] = {
            'mean_dice': float(np.mean(all_dice)) if all_dice else 0.0,
            'mean_iou': float(np.mean(all_iou)) if all_iou else 0.0
        }

        # ============ SCENARIO-BASED METRICS ============
        print("  Computing scenario-based metrics...")
        scenario_metrics = {}

        for scenario, stats in self.scenario_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_present = np.mean(stats['structures_present']) if stats['structures_present'] else 0
            avg_predicted = np.mean(stats['structures_predicted']) if stats['structures_predicted'] else 0

            scenario_metrics[scenario] = {
                'accuracy': float(accuracy),
                'num_samples': stats['total'],
                'avg_structures_present': float(avg_present),
                'avg_structures_predicted': float(avg_predicted)
            }

        metrics['scenario_based'] = scenario_metrics

        print("  Metrics calculation completed!")
        return metrics

    def create_lightweight_visualizations(self, metrics):
        """Create essential visualizations with minimal memory usage"""
        print("Creating lightweight visualizations...")

        plt.style.use('default')
        sns.set_palette("husl")

        # ============ SAMPLE VISUALIZATIONS ============
        print("  Creating sample visualizations...")
        self.visualize_selected_samples()

        # ============ PRESENCE DETECTION VISUALIZATIONS ============
        print("  Creating presence detection visualizations...")
        self.visualize_presence_metrics(metrics)

        # ============ SEGMENTATION VISUALIZATIONS ============
        print("  Creating segmentation visualizations...")
        self.visualize_segmentation_metrics(metrics)

        # ============ SCENARIO ANALYSIS ============
        print("  Creating scenario analysis...")
        self.visualize_scenario_analysis(metrics)

        # ============ CONFUSION MATRICES ============
        print("  Creating confusion matrices...")
        self.visualize_confusion_matrices()

        print("  Lightweight visualizations completed!")

    def visualize_selected_samples(self):
        """Visualize selected samples stored during evaluation"""
        if not self.visualization_samples:
            print("    No visualization samples available")
            return

        samples_per_figure = 6
        n_figures = (len(self.visualization_samples) + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(n_figures):
            start_idx = fig_idx * samples_per_figure
            end_idx = min(start_idx + samples_per_figure, len(self.visualization_samples))
            current_samples = end_idx - start_idx

            fig, axes = plt.subplots(current_samples, 3, figsize=(15, 5 * current_samples))
            if current_samples == 1:
                axes = axes.reshape(1, -1)

            for sample_idx in range(current_samples):
                sample = self.visualization_samples[start_idx + sample_idx]

                # Original image
                img_slice = np.rot90(sample['image_slice'], -1)
                axes[sample_idx, 0].imshow(img_slice, cmap='gray', vmin=0, vmax=255)
                axes[sample_idx, 0].set_title(f'Sample {sample["index"]}\nScenario: {sample["scenario"]}')
                axes[sample_idx, 0].axis('off')

                # True presence labels
                present_true = [self.structure_names[i] for i, p in enumerate(sample['true_presence']) if p == 1]
                text_true = f"Present ({len(present_true)}):\n"
                text_true += "\n".join(present_true[:8])
                if len(present_true) > 8:
                    text_true += f"\n...and {len(present_true) - 8} more"

                axes[sample_idx, 1].text(0.05, 0.95, text_true,
                                         transform=axes[sample_idx, 1].transAxes,
                                         fontsize=8, verticalalignment='top', color='green')
                axes[sample_idx, 1].set_title('True Labels')
                axes[sample_idx, 1].axis('off')

                # Predicted presence labels with confidence
                pred_binary = (sample['pred_presence'] > 0.5).astype(int)
                present_pred = [f"{self.structure_names[i]} ({sample['confidence'][i]:.2f})"
                                for i, p in enumerate(pred_binary) if p == 1]

                text_pred = f"Predicted ({len(present_pred)}):\n"
                text_pred += "\n".join(present_pred[:8])
                if len(present_pred) > 8:
                    text_pred += f"\n...and {len(present_pred) - 8} more"

                axes[sample_idx, 2].text(0.05, 0.95, text_pred,
                                         transform=axes[sample_idx, 2].transAxes,
                                         fontsize=8, verticalalignment='top', color='blue')
                axes[sample_idx, 2].set_title('Predictions (Confidence)')
                axes[sample_idx, 2].axis('off')

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

        # Support (number of positive samples)
        supports = [presence_metrics[s]['support_positive'] for s in structures]
        bars3 = axes[1, 0].bar(range(len(structures)), supports, color='gold')
        axes[1, 0].set_title('Support (Number of Positive Samples)')
        axes[1, 0].set_xlabel('Anatomical Structures')
        axes[1, 0].set_ylabel('Number of Positive Samples')
        axes[1, 0].set_xticks(range(len(structures)))
        axes[1, 0].set_xticklabels(structures, rotation=45, ha='right')

        # Confusion matrix summary
        total_tp = sum(presence_metrics[s]['true_positives'] for s in structures)
        total_fp = sum(presence_metrics[s]['false_positives'] for s in structures)
        total_fn = sum(presence_metrics[s]['false_negatives'] for s in structures)
        total_tn = sum(presence_metrics[s]['true_negatives'] for s in structures)

        summary_cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
        sns.heatmap(summary_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Absent', 'Predicted Present'],
                    yticklabels=['Actually Absent', 'Actually Present'],
                    ax=axes[1, 1])
        axes[1, 1].set_title('Overall Confusion Matrix')

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

        # IoU scores
        bars2 = axes[1].bar(x_pos, iou_means, color='lightgreen', alpha=0.8)
        axes[1].set_title('IoU Scores by Structure')
        axes[1].set_xlabel('Anatomical Structures')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(structures, rotation=45, ha='right')
        axes[1].set_ylim(0, 1)

        # Number of samples
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

        # Complexity vs accuracy
        axes[1, 0].scatter(avg_present, accuracies, s=[n * 5 for n in num_samples], alpha=0.7, c='purple')
        for i, scenario in enumerate(scenarios):
            axes[1, 0].annotate(scenario[:8], (avg_present[i], accuracies[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Average Number of Present Structures')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy vs Scenario Complexity\n(Bubble size = sample count)')

        # Summary statistics
        avg_accuracy = np.mean(accuracies)
        total_samples = sum(num_samples)

        summary_text = f"Total Scenarios: {len(scenarios)}\n"
        summary_text += f"Total Samples: {total_samples}\n"
        summary_text += f"Average Accuracy: {avg_accuracy:.3f}\n"
        summary_text += f"Best Scenario: {scenarios[np.argmax(accuracies)][:15]}\n"
        summary_text += f"Worst Scenario: {scenarios[np.argmin(accuracies)][:15]}"

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat'))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_confusion_matrices(self):
        """Create confusion matrices for each structure using accumulated data"""
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

            cm = self.presence_confusion_matrices[i]

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

    def save_lightweight_results(self, metrics):
        """Save essential results with minimal memory footprint"""
        print("Saving lightweight results...")

        # Save metrics as JSON
        with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create summary report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MEMORY-EFFICIENT MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall statistics
        report_lines.append(f"Total test samples processed: {self.n_samples_processed}")
        report_lines.append(f"Number of anatomical structures: {len(self.structure_names)}")
        report_lines.append(f"Overall presence detection accuracy: {metrics['presence_overall_accuracy']:.4f}")
        report_lines.append(f"Overall mean Dice score: {metrics['segmentation_overall']['mean_dice']:.4f}")
        report_lines.append(f"Overall mean IoU score: {metrics['segmentation_overall']['mean_iou']:.4f}")
        report_lines.append("")

        # Per-structure presence detection performance
        report_lines.append("PRESENCE DETECTION PERFORMANCE BY STRUCTURE:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Structure':<15} {'Accuracy':<10} {'F1':<8} {'Support':<8}")
        report_lines.append("-" * 60)

        for struct_name in self.structure_names:
            metrics_struct = metrics['presence_per_structure'][struct_name]
            report_lines.append(
                f"{struct_name[:14]:<15} "
                f"{metrics_struct['accuracy']:<10.4f} "
                f"{metrics_struct['f1_score']:<8.4f} "
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

        # Attention analysis summary
        if self.attention_stats:
            report_lines.append("")
            report_lines.append("ATTENTION MECHANISM ANALYSIS (First 5 samples):")
            report_lines.append("-" * 50)
            report_lines.append(f"{'Structure':<15} {'Mean Attn':<12} {'Attn Std':<12}")
            report_lines.append("-" * 50)

            for struct_name in self.structure_names:
                if struct_name in self.attention_stats:
                    stats = self.attention_stats[struct_name]
                    attn_std = np.sqrt(stats['var'])
                    report_lines.append(
                        f"{struct_name[:14]:<15} "
                        f"{stats['mean']:<12.4f} "
                        f"{attn_std:<12.4f}"
                    )

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Results saved to {self.output_dir}")


def load_model_from_checkpoint(checkpoint_path, structure_names, image_size, device):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    try:
        # Try loading with weights_only=True first (more secure)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"  Secure loading failed, trying compatibility mode...")
        print(f"  Error: {str(e)[:100]}...")
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


def main():
    """Memory-efficient main testing function"""
    parser = argparse.ArgumentParser(description='Memory-Efficient Medical Model Testing')

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
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for inference')
    parser.add_argument('--max-vis-samples', type=int, default=20,
                        help='Maximum number of samples to store for visualization')

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
    print("MEMORY-EFFICIENT MEDICAL MODEL TESTING")
    print("=" * 80)
    print(f"Weights file: {args.weights}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output}")
    print(f"Device: {device}")
    print("")

    # Load dataset metadata
    print("Loading dataset metadata...")
    try:
        with open(Path(args.dataset) / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        structure_names = metadata['structure_names']
        image_size = metadata['image_size']

        print(f"  Found {len(structure_names)} anatomical structures")
        print(f"  Image size: {image_size}")

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

    # Create dataset
    print(f"\nCreating {args.split} dataset...")
    try:
        dataset = HDF5MedicalDataset(args.dataset, split=args.split)
        print(f"  Dataset loaded: {len(dataset)} samples")

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Create memory-efficient evaluator
    print(f"\nInitializing memory-efficient evaluator...")
    evaluator = MemoryEfficientEvaluator(model, device, structure_names, args.output)
    evaluator.max_vis_samples = args.max_vis_samples

    # Run evaluation sample by sample
    print(f"\nRunning memory-efficient evaluation...")
    print("Processing samples one at a time to minimize memory usage...")

    try:
        # Process samples individually
        for idx in tqdm(range(len(dataset)), desc="Processing samples"):
            sample = dataset[idx]
            evaluator.evaluate_sample(sample)

            # Periodic garbage collection
            if idx % 100 == 0 and idx > 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        print(f"\nEvaluation completed! Processed {evaluator.n_samples_processed} samples")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Calculate metrics from accumulated data
    print("\nCalculating metrics from accumulated data...")
    try:
        metrics = evaluator.calculate_metrics()
        print("  Metrics calculation completed!")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    # Create lightweight visualizations
    print("\nCreating lightweight visualizations...")
    try:
        evaluator.create_lightweight_visualizations(metrics)
        print("  Visualizations completed!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return

    # Save results
    print("\nSaving results...")
    try:
        evaluator.save_lightweight_results(metrics)
        print("  Results saved!")
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Samples processed: {evaluator.n_samples_processed}")
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
    print("Generated files:")
    print("  - evaluation_report.txt: Detailed text report")
    print("  - detailed_metrics.json: All metrics in JSON format")
    print("  - sample_predictions_*.png: Selected sample visualizations")
    print("  - presence_detection_metrics.png: Presence detection performance")
    print("  - segmentation_metrics.png: Segmentation performance")
    print("  - scenario_analysis.png: Performance by scenario")
    print("  - confusion_matrices.png: Confusion matrices for all structures")
    print("\nMemory-efficient evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()