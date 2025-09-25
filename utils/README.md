# Synthetic Medical Dataset Generator

A comprehensive Python tool for generating synthetic 3D medical imaging datasets with anatomical structures, designed for testing and training medical AI models. The generator creates realistic CT/MRI-like images with configurable anatomical structures and pathological conditions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Usage Examples](#usage-examples)
- [Dataset Structure](#dataset-structure)
- [Anatomical Structures](#anatomical-structures)
- [Medical Scenarios](#medical-scenarios)
- [Data Loading](#data-loading)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

This tool generates synthetic 3D medical images containing various anatomical structures with realistic imaging artifacts. It's designed to:

- **Train medical AI models** when real data is limited or restricted
- **Test model robustness** across different anatomical variations
- **Benchmark algorithms** on controlled datasets
- **Prototype medical imaging applications** quickly
- **Educational purposes** for medical imaging courses

The generator creates datasets in efficient HDF5 format with lazy loading capabilities for memory-efficient training of deep learning models.

## Features

### Core Capabilities
- ✅ **3D Medical Image Generation** - Realistic CT/MRI-like volumes
- ✅ **15+ Anatomical Structures** - Organs, bones, and soft tissues
- ✅ **Pathological Scenarios** - Missing organs, surgical conditions
- ✅ **Realistic Imaging Artifacts** - Noise, bias fields, resolution effects
- ✅ **Flexible Image Sizes** - From 64³ to 512³+ voxels
- ✅ **Memory-Efficient Storage** - Compressed HDF5 with chunking
- ✅ **PyTorch Integration** - Ready-to-use Dataset classes
- ✅ **Reproducible Generation** - Seed-controlled randomness

### Medical Realism
- **Anatomically Correct Positioning** - Based on real human anatomy
- **Organ Size Variations** - Natural biological variation
- **Imaging Physics** - Blur, bias fields, and noise modeling
- **Clinical Scenarios** - Post-surgical, congenital, and traumatic conditions

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy matplotlib pathlib h5py
pip install scikit-learn torch torchvision
pip install dataclasses typing

# Optional for enhanced functionality
pip install tqdm  # Progress bars
pip install pillow  # Additional image processing
```

### Installation

```bash
# Clone or download the script
wget https://your-repo/synthetic_medical_generator.py

# Or copy the complete script to your project
cp synthetic_medical_generator.py your_project/
```

## Quick Start

### Basic Dataset Generation

```bash
# Generate a standard dataset
python synthetic_medical_generator.py \
    --image-size "128,128,64" \
    --total-samples 1000 \
    --output-dir "./medical_dataset"
```

### Load and Visualize

```python
from synthetic_medical_generator import LazyMedicalDataset, visualize_sample

# Load dataset
dataset = LazyMedicalDataset('./medical_dataset', split='train')

# Visualize a sample
visualize_sample(dataset, idx=0)

# Check dataset statistics
distribution = dataset.get_class_distribution()
print(f"Dataset contains {len(dataset)} samples")
```

## Command Line Interface

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image-size` | str | "128,128,64" | 3D image dimensions as "Z,Y,X" |
| `--total-samples` | int | 1000 | Total number of samples to generate |
| `--output-dir` | str | "../data/synthetic_medical_dataset_hdf5" | Output directory path |

### Dataset Split Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--validation-split` | float | 0.2 | Fraction for validation (0.0-1.0) |
| `--test-split` | float | 0.1 | Fraction for testing (0.0-1.0) |

### Realism Control Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--position-noise` | float | 0.05 | Standard deviation for organ position variation |
| `--size-noise` | float | 0.2 | Standard deviation for organ size variation |
| `--intensity-noise` | float | 0.15 | Standard deviation for intensity variation |

### Control Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip-generation` | flag | False | Skip dataset creation (test loading only) |
| `--skip-visualization` | flag | False | Skip sample visualization |
| `--skip-training-test` | flag | False | Skip training loop demonstration |
| `--verbose` | flag | False | Enable detailed output |
| `--seed` | int | 42 | Random seed for reproducibility |

### Help and Information

```bash
# Display all options
python synthetic_medical_generator.py --help

# Show version info
python synthetic_medical_generator.py --version  # If implemented
```

## Usage Examples

### Basic Usage

```bash
# Standard dataset generation
python synthetic_medical_generator.py
```

### Custom Image Sizes

```bash
# High resolution dataset (warning: memory intensive)
python synthetic_medical_generator.py \
    --image-size "256,256,128" \
    --total-samples 500 \
    --output-dir "./high_res_dataset"

# Low resolution for quick prototyping
python synthetic_medical_generator.py \
    --image-size "64,64,32" \
    --total-samples 2000 \
    --output-dir "./quick_dataset"

# Ultra-high resolution (requires significant RAM)
python synthetic_medical_generator.py \
    --image-size "512,512,256" \
    --total-samples 100 \
    --output-dir "./ultra_hires"
```

### Dataset Size Variations

```bash
# Large training dataset
python synthetic_medical_generator.py \
    --total-samples 10000 \
    --validation-split 0.15 \
    --test-split 0.05

# Small dataset for testing
python synthetic_medical_generator.py \
    --total-samples 100 \
    --validation-split 0.3 \
    --test-split 0.2

# Training-only dataset (no test split)
python synthetic_medical_generator.py \
    --total-samples 5000 \
    --test-split 0.0
```

### Realism Control

```bash
# High variation (more challenging dataset)
python synthetic_medical_generator.py \
    --position-noise 0.1 \
    --size-noise 0.4 \
    --intensity-noise 0.25

# Low variation (more consistent)
python synthetic_medical_generator.py \
    --position-noise 0.02 \
    --size-noise 0.1 \
    --intensity-noise 0.05

# Extreme variation for robustness testing
python synthetic_medical_generator.py \
    --position-noise 0.15 \
    --size-noise 0.5 \
    --intensity-noise 0.3
```

### Development and Testing

```bash
# Test existing dataset without regeneration
python synthetic_medical_generator.py \
    --skip-generation \
    --output-dir "./existing_dataset"

# Generate without visualization (faster)
python synthetic_medical_generator.py \
    --skip-visualization \
    --skip-training-test \
    --verbose

# Debug mode with detailed output
python synthetic_medical_generator.py \
    --verbose \
    --total-samples 50 \
    --image-size "64,64,32"
```

### Reproducibility

```bash
# Reproducible dataset with custom seed
python synthetic_medical_generator.py \
    --seed 12345 \
    --total-samples 1000

# Multiple datasets with different seeds
for seed in {1..5}; do
    python synthetic_medical_generator.py \
        --seed $seed \
        --output-dir "./dataset_seed_$seed" \
        --total-samples 500
done
```

### Production Workflows

```bash
# Large-scale production dataset
python synthetic_medical_generator.py \
    --image-size "256,256,128" \
    --total-samples 50000 \
    --validation-split 0.1 \
    --test-split 0.05 \
    --output-dir "/data/production_medical_dataset" \
    --skip-visualization \
    --skip-training-test \
    --seed 42

# Multi-resolution dataset generation
resolutions=("64,64,32" "128,128,64" "256,256,128")
for res in "${resolutions[@]}"; do
    python synthetic_medical_generator.py \
        --image-size "$res" \
        --total-samples 2000 \
        --output-dir "./dataset_${res//,/x}" \
        --skip-visualization
done
```

## Dataset Structure

### Directory Layout

```
output_dir/
├── train.h5              # Training data (HDF5 format)
├── val.h5                # Validation data
├── test.h5               # Test data (if test_split > 0)
└── metadata.json         # Dataset metadata and configuration
```

### HDF5 File Structure

```
train.h5
├── images                # Shape: [N, Z, Y, X] - 3D medical images
├── masks                 # Shape: [N, n_structures, Z, Y, X] - Binary segmentation masks
├── presence_labels       # Shape: [N, n_structures] - Binary presence indicators
├── scenarios            # Shape: [N] - Scenario names (string array)
└── attributes           # HDF5 attributes with metadata
```

### Metadata Format

```json
{
  "structure_names": [
    "left_lung", "right_lung", "heart", "liver", 
    "spleen", "pancreas", "left_kidney", "right_kidney",
    "L1", "L2", "L3", "L4", "L5"
  ],
  "image_size": [128, 128, 64],
  "generation_params": {
    "position_noise": 0.05,
    "size_noise": 0.2,
    "intensity_noise": 0.15
  },
  "scenarios": [...]
}
```

## Anatomical Structures

The generator includes 13+ anatomical structures based on real human anatomy:

### Thoracic Structures

| Structure | Location | Size (relative) | Intensity | Shape |
|-----------|----------|-----------------|-----------|-------|
| **Left Lung** | Left thorax | 0.15×0.25×0.35 | 150 | Ellipsoid |
| **Right Lung** | Right thorax | 0.18×0.25×0.35 | 150 | Ellipsoid |
| **Heart** | Central thorax | 0.12×0.15×0.2 | 200 | Ellipsoid |

### Abdominal Structures

| Structure | Location | Size (relative) | Intensity | Shape |
|-----------|----------|-----------------|-----------|-------|
| **Liver** | Right upper abdomen | 0.25×0.2×0.3 | 180 | Ellipsoid |
| **Spleen** | Left upper abdomen | 0.08×0.1×0.15 | 170 | Ellipsoid |
| **Pancreas** | Central abdomen | 0.15×0.04×0.08 | 160 | Ellipsoid |
| **Left Kidney** | Left retroperitoneum | 0.08×0.12×0.15 | 160 | Kidney-shaped |
| **Right Kidney** | Right retroperitoneum | 0.08×0.12×0.15 | 160 | Kidney-shaped |

### Skeletal Structures

| Structure | Location | Size (relative) | Intensity | Shape |
|-----------|----------|-----------------|-----------|-------|
| **L1-L5 Vertebrae** | Posterior spine | 0.06×0.06×0.08 | 220 | Cylindrical |

### Imaging Characteristics

- **Intensity Values**: Scaled to simulate HU (Hounsfield Units) in CT
- **Spatial Relationships**: Anatomically accurate positioning
- **Size Variations**: Natural biological variation (±20%)
- **Imaging Artifacts**: Blur, noise, and bias field effects

## Medical Scenarios

The generator creates diverse medical scenarios to test AI model robustness:

### Normal Anatomy (40% of dataset)

- **normal_complete**: All structures present
- **normal_thoracic**: Thoracic structures only
- **normal_abdominal**: Abdominal structures only

### Surgical Conditions (35% of dataset)

| Scenario | Description | Clinical Context |
|----------|-------------|------------------|
| **left_nephrectomy** | Left kidney removed | Kidney cancer, trauma |
| **right_nephrectomy** | Right kidney removed | Kidney donation, disease |
| **splenectomy** | Spleen removed | Trauma, blood disorders |
| **bilateral_nephrectomy** | Both kidneys removed | End-stage renal disease |
| **left_lung_resection** | Left lung removed | Lung cancer |

### Traumatic/Congenital (20% of dataset)

| Scenario | Description | Clinical Context |
|----------|-------------|------------------|
| **missing_L4** | L4 vertebra absent | Spina bifida, trauma |
| **missing_L2_L3** | Multiple vertebrae absent | Congenital anomaly |
| **complex_abdominal_surgery** | Multiple organ removal | Complex surgical case |

### Edge Cases (5% of dataset)

- **minimal_structures**: Very few organs present
- **extensive_resection**: Multiple major organs removed

## Data Loading

### PyTorch Integration

```python
from synthetic_medical_generator import LazyMedicalDataset
from torch.utils.data import DataLoader

# Create datasets
train_dataset = LazyMedicalDataset(
    'path/to/dataset', 
    split='train',
    load_masks=True,  # Set False for classification only
    transform=your_transform_function
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for batch in train_loader:
    images = batch['image']          # [B, Z, Y, X]
    masks = batch['masks']           # [B, n_structures, Z, Y, X] 
    presence = batch['presence_labels']  # [B, n_structures]
    scenarios = batch['scenario']    # List of scenario names
    
    # Your training code here
```

### Memory-Efficient Loading

```python
# Load only what you need
classification_dataset = LazyMedicalDataset(
    'path/to/dataset',
    split='train',
    load_masks=False,  # Save memory for classification tasks
    preload_metadata=True  # Faster scenario access
)

# Custom transforms
def medical_transform(image, masks=None):
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Extract 2D slices for 2D models
    # slice_idx = image.shape[-1] // 2
    # image = image[..., slice_idx]
    
    # Add channel dimension if needed
    # image = image[..., None]
    
    if masks is not None:
        return image, masks
    return image

dataset = LazyMedicalDataset(
    'path/to/dataset',
    transform=medical_transform
)
```

### Dataset Analysis

```python
# Get dataset statistics
dataset = LazyMedicalDataset('path/to/dataset', split='train')

print(f"Dataset size: {len(dataset)}")
print(f"Image shape: {dataset.image_size}")
print(f"Structures: {dataset.get_structure_names()}")

# Class distribution
distribution = dataset.get_class_distribution()
for structure, stats in distribution.items():
    print(f"{structure}: {stats['presence_rate']:.1%} present")

# Sample inspection
sample = dataset[0]
print(f"Sample scenario: {sample['scenario']}")
print(f"Structures present: {sample['presence_labels'].sum().item()}")
```

## API Reference

### Core Classes

#### `SyntheticAnatomyGenerator`
Primary generator for individual medical images.

```python
generator = SyntheticAnatomyGenerator(image_size=(128, 128, 64))

# Generate single sample
image, masks, presence = generator.generate_sample(
    present_structures=['heart', 'liver', 'left_lung'],
    position_noise=0.05,
    size_noise=0.2,
    intensity_noise=0.15
)
```

#### `SyntheticDatasetGenerator` 
High-level dataset generation with scenarios.

```python
dataset_gen = SyntheticDatasetGenerator(image_size=(128, 128, 64))

# Generate complete dataset
dataset = dataset_gen.generate_dataset(
    total_samples=1000,
    validation_split=0.2,
    test_split=0.1
)

# Save in HDF5 format
dataset_gen.save_dataset_hdf5(dataset, 'output/path')
```

#### `LazyMedicalDataset`
PyTorch Dataset for memory-efficient loading.

```python
dataset = LazyMedicalDataset(
    data_path='path/to/hdf5/dataset',
    split='train',
    transform=None,
    load_masks=True,
    preload_metadata=True
)
```

### Key Methods

#### Dataset Generation
- `generate_sample()`: Create single medical image
- `generate_dataset()`: Create complete train/val/test splits
- `save_dataset_hdf5()`: Save in compressed HDF5 format

#### Data Loading
- `__getitem__()`: Load individual samples
- `get_class_distribution()`: Analyze label distribution
- `get_structure_names()`: List available structures

#### Visualization
- `visualize_sample()`: Display image with masks and labels

## Performance Considerations

### Memory Usage

| Image Size | Single Image | 1000 Images | Recommended RAM |
|------------|--------------|-------------|-----------------|
| 64×64×32 | 0.5 MB | 500 MB | 4 GB |
| 128×128×64 | 4 MB | 4 GB | 16 GB |
| 256×256×128 | 32 MB | 32 GB | 64 GB |
| 512×512×256 | 256 MB | 256 GB | 512 GB |

### Generation Time

| Image Size | Per Sample | 1000 Samples | Hardware |
|------------|------------|--------------|-----------|
| 64×64×32 | 0.1s | 2 min | CPU |
| 128×128×64 | 0.5s | 8 min | CPU |
| 256×256×128 | 2s | 33 min | CPU |
| 512×512×256 | 8s | 2.2 hrs | CPU |

### Optimization Tips

```bash
# Use appropriate image size for your task
--image-size "128,128,64"  # Good balance for most tasks

# Generate in batches for large datasets
--total-samples 10000 --skip-visualization --skip-training-test

# Use HDF5 chunking for efficient random access
# (automatically handled by the generator)

# Adjust worker processes for data loading
num_workers=4  # In DataLoader, match your CPU cores
```

### Storage Requirements

```bash
# Compressed HDF5 sizes (approximate)
64³ × 1000 samples:   ~200 MB
128³ × 1000 samples:  ~800 MB  
256³ × 1000 samples:  ~3.2 GB
512³ × 1000 samples:  ~13 GB
```

## Troubleshooting

### Common Issues

#### Memory Errors

```bash
# Error: Out of memory during generation
# Solution: Reduce image size or sample count
python synthetic_medical_generator.py --image-size "64,64,32" --total-samples 500

# Error: Out of memory during loading
# Solution: Reduce batch size and use fewer workers
train_loader = DataLoader(dataset, batch_size=2, num_workers=1)
```

#### File System Issues

```bash
# Error: Permission denied
# Solution: Check write permissions
chmod 755 /path/to/output/directory

# Error: Disk space
# Solution: Check available space
df -h /path/to/output/directory
```

#### Performance Issues

```bash
# Slow generation
# Solution: Use smaller images for prototyping
--image-size "64,64,32" --total-samples 100

# Slow data loading
# Solution: Reduce workers or enable persistent workers
persistent_workers=True  # In DataLoader
```

### Validation

```python
# Verify dataset integrity
dataset = LazyMedicalDataset('path/to/dataset')
print(f"All samples loadable: {all(dataset[i] is not None for i in range(len(dataset)))}")

# Check for corruption
import h5py
with h5py.File('path/to/dataset/train.h5', 'r') as f:
    print(f"Images shape: {f['images'].shape}")
    print(f"Masks shape: {f['masks'].shape}")
    print(f"Labels shape: {f['presence_labels'].shape}")
```

### Getting Help

1. **Check verbose output**: Use `--verbose` flag for detailed information
2. **Validate parameters**: Ensure splits sum to < 1.0, positive sample counts
3. **Monitor resources**: Check RAM and disk space during generation
4. **Test with small datasets**: Use `--total-samples 10` for quick validation

For additional support, refer to the inline code documentation and error messages, which provide specific guidance for common issues.