# SADAAN Medical Segmentation Training Documentation

## Overview

This training script implements a complete pipeline for training the SADAAN (Spatial Attention Deep Anatomical Network) model on medical imaging data. It features memory-efficient HDF5 dataset loading, comprehensive experiment tracking with Sacred, and optimized training procedures for 3D medical image segmentation.

## Key Features

- **Sacred Experiment Tracking**: Comprehensive logging and reproducibility
- **Multi-GPU Support**: CUDA acceleration with memory optimization
- **Data Augmentation**: Medical-specific augmentations (noise, intensity scaling)
- **Early Stopping & Checkpointing**: Robust training with automatic model saving

## Requirements

### Dependencies
```bash
pip install torch torchvision
pip install sacred pymongo
pip install h5py
pip install numpy matplotlib tqdm
pip install pathlib
```

## Dataset Format

The script expects HDF5 datasets with the following structure:

```
synthetic_medical_dataset_hdf5/
├── metadata.json          # Dataset metadata
├── train.h5              # Training data
├── val.h5                # Validation data
└── test.h5               # Test data (optional)
```

### HDF5 File Structure
Each `.h5` file contains:
- `images`: 4D array [N, H, W, D] - Medical images
- `masks`: 5D array [N, S, H, W, D] - Segmentation masks per structure
- `presence_labels`: 2D array [N, S] - Binary presence indicators
- `scenarios`: 1D array [N] - Scenario descriptions

### Metadata Format
```json
{
    "structure_names": ["liver", "kidney", "spleen"],
    "image_size": [128, 128, 64],
    "num_structures": 3,
    "dataset_info": "Medical segmentation dataset"
}
```

## Command Line Usage

### Basic Training
```bash
# Train with default parameters
python train_hdf5.py

# Specify dataset path
python train_hdf5.py with dataset_path='./my_medical_dataset_hdf5'
```

### Training Configuration
```bash
# Adjust batch size and learning rate
python train_hdf5.py with training.batch_size=8 training.learning_rate=1e-3

# Extended training with early stopping
python train_hdf5.py with training.num_epochs=200 training.early_stopping_patience=30

# Memory-constrained training
python train_hdf5.py with training.batch_size=2 training.gradient_accumulation_steps=4
```

### Model Configuration
```bash
# Single-channel input (CT/MRI)
python train_hdf5.py with model.in_channels=1

# Multi-channel input
python train_hdf5.py with model.in_channels=3

# Adjust presence threshold
python train_hdf5.py with model.presence_threshold=0.3
```

### Loss Function Tuning
```bash
# Emphasize dice loss
python train_hdf5.py with loss_weights.dice=3.0 loss_weights.segmentation=0.5

# Focus on attention supervision
python train_hdf5.py with loss_weights.attention_supervision=1.0

# Reduce false positives
python train_hdf5.py with loss_weights.false_positive_suppression=1.0
```

### Data Augmentation
```bash
# Enable augmentation with custom parameters
python train_hdf5.py with augmentation.enabled=True augmentation.noise_std=3.0

# Intensity scaling range
python train_hdf5.py with augmentation.intensity_scale_range=[0.9,1.1]

# Disable augmentation
python train_hdf5.py with augmentation.enabled=False
```

### Optimizer Settings
```bash
# AdamW with custom weight decay
python train_hdf5.py with optimizer.type='adamw' optimizer.weight_decay=5e-4

# Cosine annealing scheduler
python train_hdf5.py with optimizer.lr_scheduler='cosine_annealing'

# Reduce on plateau
python train_hdf5.py with optimizer.lr_scheduler='reduce_on_plateau'
```

### Checkpoint Management
```bash
# Custom checkpoint directory
python train_hdf5.py with checkpoint.save_dir='./experiments/run_001'

# Frequent checkpointing
python train_hdf5.py with checkpoint.save_frequency=5

# Keep only best model
python train_hdf5.py with checkpoint.keep_best_only=True
```

### Advanced Usage Examples

#### High-Memory Training
```bash
python train_hdf5.py with \
    training.batch_size=16 \
    training.num_workers=8 \
    training.image_size=[256,256,128] \
    training.learning_rate=1e-3
```

#### Memory-Constrained Training
```bash
python train_hdf5.py with \
    training.batch_size=2 \
    training.gradient_accumulation_steps=8 \
    training.memory_cleanup_frequency=2 \
    training.num_workers=2
```

#### Research Configuration
```bash
python train_hdf5.py with \
    training.num_epochs=500 \
    training.early_stopping_patience=50 \
    loss_weights.attention_supervision=1.5 \
    optimizer.lr_scheduler='cosine_annealing' \
    augmentation.enabled=True \
    logging.visualize_predictions=True
```

#### Quick Debugging Run
```bash
python train_hdf5.py with \
    training.num_epochs=5 \
    training.batch_size=2 \
    logging.log_frequency=1 \
    checkpoint.save_frequency=1
```

## Configuration Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Training batch size |
| `image_size` | [128,128,64] | 3D image dimensions |
| `learning_rate` | 5e-4 | Initial learning rate |
| `num_epochs` | 100 | Maximum training epochs |
| `early_stopping_patience` | 20 | Epochs to wait before stopping |
| `gradient_clip_norm` | 0.5 | Gradient clipping threshold |
| `num_workers` | 2 | DataLoader worker processes |
| `warmup_epochs` | 5 | Learning rate warmup period |
| `gradient_accumulation_steps` | 2 | Steps before optimizer update |

### Loss Weights
| Weight | Default | Purpose |
|--------|---------|---------|
| `segmentation` | 1.0 | Base segmentation loss |
| `dice` | 2.0 | Dice coefficient loss |
| `focal_seg` | 1.0 | Focal loss for hard examples |
| `absence` | 1.0 | Absence detection loss |
| `attention_supervision` | 0.5 | Attention map supervision |
| `false_positive_suppression` | 0.5 | FP reduction |
| `confidence` | 0.1 | Confidence regularization |

### Augmentation Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | True | Enable data augmentation |
| `noise_std` | 2.0 | Gaussian noise standard deviation |
| `intensity_scale_range` | [0.95,1.05] | Random intensity scaling |
| `rotation_degrees` | 5 | Random rotation range |
| `flip_probability` | 0.3 | Random flip probability |

## Output and Logging

### Sacred Experiment Tracking
The script automatically logs:
- **Metrics**: Loss values, accuracy, Dice scores
- **Parameters**: All configuration parameters
- **Artifacts**: Model checkpoints, training plots
- **System Info**: Hardware, software versions

### File Outputs
```
checkpoints/
├── best_model.pth           # Best validation performance
├── checkpoint_epoch_10.pth  # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── ...

sacred_runs/
├── 1/                      # Experiment run directory
│   ├── config.json         # Configuration used
│   ├── run.json           # Run metadata
│   ├── metrics.json       # Logged metrics
│   └── cout.txt          # Console output
```

### Metrics Logged
- **Training/Validation Loss**: Total and component losses
- **Presence Detection**: Accuracy per structure
- **Segmentation Quality**: Dice coefficients
- **Learning Rate**: Current optimizer LR
- **System Metrics**: GPU memory usage

## Monitoring Training

### Sacred Web Interface
```bash
# Start Sacred web interface
sacredboard -m localhost:27017:sadaan
```

### TensorBoard Alternative
The script logs to Sacred, but you can extend it for TensorBoard:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
```

## Memory Optimization

The script includes several memory optimization strategies:

### Automatic Memory Management
- Periodic garbage collection
- CUDA cache clearing
- Gradient accumulation for large effective batch sizes

### Memory-Efficient Loading
```python
# HDF5 datasets load samples on-demand
# No need to load entire dataset into RAM
dataset = HDF5MedicalDataset(data_path, split='train')
```

### GPU Memory Tips
- Use `training.batch_size=2` and `gradient_accumulation_steps=8` for 16 effective batch size
- Enable `pin_memory=True` for faster GPU transfer
- Use `num_workers=0` if experiencing memory issues

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size and use gradient accumulation
python train_hdf5.py with training.batch_size=1 training.gradient_accumulation_steps=8
```

#### Dataset Not Found
```bash
# Verify dataset path and structure
python -c "
import h5py
f = h5py.File('./synthetic_medical_dataset_hdf5/train.h5', 'r')
print(list(f.keys()))
"
```

#### Sacred MongoDB Issues
```bash
# Use file observer instead
python train_hdf5.py  # Uses FileStorageObserver by default
```

#### Slow Data Loading
```bash
# Increase workers and enable pin_memory
python train_hdf5.py with training.num_workers=4
```

### Performance Tuning

#### For Speed
```bash
python train_hdf5.py with \
    training.num_workers=8 \
    training.memory_cleanup_frequency=10 \
    augmentation.enabled=False
```

#### For Accuracy
```bash
python train_hdf5.py with \
    loss_weights.dice=3.0 \
    training.gradient_clip_norm=1.0 \
    optimizer.weight_decay=1e-5 \
    augmentation.noise_std=1.0
```

## Model Architecture

The SADAAN model combines:
- **3D Spatial Attention**: Focuses on relevant anatomical regions
- **Multi-Scale Features**: Captures details at different resolutions
- **Presence Detection**: Determines if structures are present
- **Joint Training**: Simultaneous segmentation and presence prediction

### Key Components
1. **Encoder**: Feature extraction from 3D medical images
2. **Spatial Attention**: Learned attention maps
3. **Decoder**: Segmentation mask generation
4. **Presence Head**: Structure presence classification

## Advanced Usage

### Custom Loss Weights
```python
# In your experiment
python train_hdf5.py with loss_weights='{
    "segmentation": 1.0,
    "dice": 4.0,
    "focal_seg": 0.5,
    "attention_supervision": 2.0
}'
```

### Multiple GPU Training
```bash
# The script auto-detects available GPUs
CUDA_VISIBLE_DEVICES=0,1 python train_hdf5.py with training.batch_size=8
```

### Resume Training
To resume from a checkpoint, modify the script to load:
```python
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```