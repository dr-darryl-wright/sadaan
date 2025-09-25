# Medical Model Testing Script Documentation

## Overview

This script provides memory-efficient evaluation and testing capabilities for medical image segmentation models using spatial attention mechanisms. It processes 3D medical images to detect anatomical structures and generate segmentation masks, with comprehensive visualization and analysis outputs.

## Features

- **Memory-efficient processing**: Processes samples individually to minimize memory usage
- **Dual-task evaluation**: Tests both presence detection and segmentation accuracy
- **Comprehensive visualizations**: Generates attention maps, segmentation comparisons, and performance metrics
- **Scenario-based analysis**: Evaluates performance across different clinical scenarios
- **Streaming metrics**: Calculates statistics without storing all data in memory

## Requirements

### Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib seaborn pandas
pip install scikit-learn h5py tqdm
pip install pathlib argparse collections
```

### Model Requirements

- SpatialAttentionMedicalSegmenter model class
- SpatialAttentionLoss (imported from `../model/sadaan.py`)
- Model checkpoint file (.pth format)

### Dataset Requirements

- HDF5 dataset with the following structure:
  ```
  dataset_directory/
  ├── metadata.json          # Contains structure names and image dimensions
  ├── train.h5               # Training data (optional)
  ├── val.h5                 # Validation data (optional)
  └── test.h5                # Test data
  ```

## Command Line Usage

### Basic Usage

```bash
python test_model.py --weights model_checkpoint.pth --dataset ./dataset_path
```

### Full Parameter Usage

```bash
python test_model.py \
  --weights ./checkpoints/best_model.pth \
  --dataset ./synthetic_medical_dataset_hdf5 \
  --split test \
  --output ./evaluation_results \
  --device cuda \
  --max-vis-samples 20
```

### Parameters

| Parameter | Short | Required | Default | Description |
|-----------|--------|----------|---------|-------------|
| `--weights` | `-w` | Yes | - | Path to model checkpoint (.pth file) |
| `--dataset` | `-d` | No | `./synthetic_medical_dataset_hdf5` | Path to HDF5 dataset directory |
| `--split` | `-s` | No | `test` | Dataset split to evaluate (`train`, `val`, `test`) |
| `--output` | `-o` | No | `./test_results` | Output directory for results |
| `--device` | - | No | `cuda` | Device for inference (`cuda`, `cpu`, `auto`) |
| `--max-vis-samples` | - | No | `20` | Maximum samples to store for detailed visualization |

## Usage Examples

### Example 1: Basic Model Testing

```bash
# Test a trained model on the default test set
python test_model.py --weights trained_model.pth
```

This will:
- Load the model from `trained_model.pth`
- Use the dataset in `./synthetic_medical_dataset_hdf5`
- Evaluate on the test split
- Save results to `./test_results`

### Example 2: Custom Dataset and Output Directory

```bash
# Test with custom dataset path and output directory
python test_model.py \
  --weights ./models/checkpoint_epoch_50.pth \
  --dataset /data/medical_scans/processed \
  --output ./evaluation_epoch_50 \
  --device cuda
```

### Example 3: CPU Testing with Limited Visualization

```bash
# Run evaluation on CPU with fewer visualization samples
python test_model.py \
  --weights model.pth \
  --dataset ./dataset \
  --device cpu \
  --max-vis-samples 5 \
  --output ./cpu_results
```

### Example 4: Validation Set Evaluation

```bash
# Evaluate on validation set instead of test set
python test_model.py \
  --weights best_model.pth \
  --split val \
  --output ./validation_results
```

### Example 5: Comprehensive Evaluation Pipeline

```bash
# Complete evaluation with custom settings
python test_model.py \
  --weights ./checkpoints/spatial_attention_model.pth \
  --dataset /mnt/medical_data/hdf5_dataset \
  --split test \
  --output ./final_evaluation \
  --device auto \
  --max-vis-samples 50
```

## Output Files

The script generates several types of output files:

### Reports and Metrics

- **`evaluation_report.txt`**: Comprehensive text report with all metrics
- **`detailed_metrics.json`**: All metrics in JSON format for further processing

### Visualizations

- **`comprehensive_analysis_sample_*.png`**: Detailed single-sample analysis including:
  - Original images (axial, coronal, sagittal views)
  - Attention maps for each detected structure
  - Position priors visualization
  - True vs predicted segmentation masks
  - Segmentation difference maps with TP/FP/FN regions

- **`attention_maps_overview.png`**: Overview of attention patterns across multiple samples

- **`segmentation_analysis_sample_*.png`**: Detailed segmentation comparisons

- **`sample_predictions_*.png`**: Basic prediction visualizations

### Performance Analysis

- **`presence_detection_metrics.png`**: Presence detection performance charts
- **`segmentation_metrics.png`**: Segmentation quality metrics
- **`scenario_analysis.png`**: Performance breakdown by clinical scenario
- **`confusion_matrices.png`**: Confusion matrices for all anatomical structures

## Dataset Format

### Metadata JSON Structure

```json
{
  "structure_names": [
    "heart", "lung_left", "lung_right", "liver", "kidney_left", "kidney_right"
  ],
  "image_size": [256, 256, 128],
  "num_structures": 6,
  "dataset_info": {
    "created": "2024-01-01",
    "version": "1.0"
  }
}
```

### HDF5 File Structure

Each split file (train.h5, val.h5, test.h5) should contain:

- **`images`**: 4D array of shape `[N, H, W, D]` - Medical image volumes
- **`masks`**: 5D array of shape `[N, num_structures, H, W, D]` - Segmentation masks
- **`presence_labels`**: 2D array of shape `[N, num_structures]` - Binary presence labels
- **`scenarios`**: 1D array of strings - Clinical scenario labels
- **Attributes**:
  - `n_samples`: Total number of samples in the file

## Performance Metrics

The script evaluates models on multiple criteria:

### Presence Detection Metrics

- **Accuracy**: Overall classification accuracy per structure
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the precision-recall curve

### Segmentation Metrics

- **Dice Coefficient**: Overlap similarity between predicted and true masks
- **IoU (Intersection over Union)**: Intersection / Union of predicted and true regions
- **Per-structure statistics**: Mean and standard deviation for each anatomical structure

### Scenario-based Analysis

- **Accuracy by scenario**: Performance breakdown across different clinical contexts
- **Complexity analysis**: Relationship between number of present structures and accuracy
- **Sample distribution**: Number of samples per scenario type

## Memory Management

The script implements several memory optimization strategies:

### Streaming Processing
- Processes one sample at a time
- Immediately moves data from GPU to CPU after processing
- Uses accumulator variables instead of storing all predictions

### Selective Visualization Storage
- Limits the number of samples stored for detailed visualization
- Compresses visualization data by storing only present structures
- Periodic garbage collection every 50 samples

### Lightweight Metrics Calculation
- Calculates running statistics for attention maps
- Uses confusion matrices for presence detection metrics
- Stores only essential data for final ROC/PR curve calculations

## Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce visualization samples
python test_model.py --weights model.pth --max-vis-samples 5

# Force CPU usage
python test_model.py --weights model.pth --device cpu
```

**Missing Dependencies**
```bash
# Install required packages
pip install torch torchvision matplotlib seaborn scikit-learn h5py tqdm pandas
```

**CUDA Issues**
```bash
# Use automatic device selection
python test_model.py --weights model.pth --device auto

# Force CPU if CUDA is problematic
python test_model.py --weights model.pth --device cpu
```

**Dataset Format Errors**
- Ensure `metadata.json` exists in dataset directory
- Verify HDF5 file structure matches expected format
- Check that structure names in metadata match model expectations

### Performance Tips

1. **GPU Memory**: Use `--max-vis-samples` to control memory usage
2. **Large Datasets**: Process in batches or use CPU for very large evaluations
3. **Storage**: Ensure sufficient disk space for visualization outputs
4. **Speed**: Use GPU when available for faster inference

## Integration Examples

### Batch Processing Multiple Models

```bash
#!/bin/bash
# Evaluate multiple model checkpoints

for epoch in 10 20 30 40 50; do
  echo "Evaluating epoch $epoch..."
  python test_model.py \
    --weights ./checkpoints/model_epoch_${epoch}.pth \
    --output ./results/epoch_${epoch} \
    --max-vis-samples 10
done
```

### Cross-validation Evaluation

```bash
#!/bin/bash
# Evaluate on different data splits

for split in train val test; do
  echo "Evaluating on $split set..."
  python test_model.py \
    --weights best_model.pth \
    --split $split \
    --output ./cv_results/$split
done
```

### Automated Model Comparison

```bash
#!/bin/bash
# Compare different model architectures

models=("attention_model.pth" "baseline_model.pth" "enhanced_model.pth")

for model in "${models[@]}"; do
  model_name=$(basename "$model" .pth)
  echo "Evaluating $model_name..."
  python test_model.py \
    --weights "./models/$model" \
    --output "./comparison/$model_name" \
    --max-vis-samples 15
done
```
