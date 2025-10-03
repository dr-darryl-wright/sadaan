# SADAAn (Spatial Attention for Detection of Absent Anatomy)

## Overview

The Spatial Attention for Detection of Absent Anatomy Model is a PyTorch-based deep learning architecture designed for multi-organ segmentation in medical imaging with built-in absence detection capabilities. The model combines U-Net-style feature extraction with anatomical attention mechanisms and presence/absence detection to handle cases where anatomical structures may be missing from scans.

## Architecture Components

### Core Model: SpatialAttentionMedicalSegmenter

The main model integrates four key components:

1. **Encoder**: U-Net style encoder-decoder with skip connections
2. **Anatomical Attention Module**: Learns spatial attention maps for anatomical structures
3. **Absence Detection Head**: Determines structure presence/absence
4. **Segmentation Head**: Produces structure-specific segmentation masks

**Key Features:**
- Handles variable presence of anatomical structures
- Uses learned anatomical position priors
- Attention-modulated feature processing
- Confidence estimation for predictions

### 1. Encoder

A U-Net style encoder-decoder network with skip connections for feature extraction.

**Architecture:**
- 5 encoder levels with progressively increasing channels (16 → 32 → 64 → 128 → 256)
- 4 decoder levels with symmetric upsampling and skip connections
- Each level uses double convolution blocks with BatchNorm and ReLU
- Final output: 128-channel feature maps at original resolution

### 2. AnatomicalAttentionModule

Learns structure-specific spatial attention based on anatomical priors.

**Parameters:**
- `in_channels`: Number of input feature channels
- `num_structures`: Number of anatomical structures to segment
- `spatial_dims`: Tuple of (depth, height, width) dimensions

**Components:**
- Learnable position embeddings for anatomical priors
- Attention convolution network for spatial attention maps
- Feature matcher for structure-specific responses
- Alignment score computation for presence detection

**Outputs:**
- `attention_maps`: Structure-specific spatial attention [B, num_structures, D, H, W]
- `feature_responses`: Feature responses for each structure
- `alignment_scores`: Attention-feature alignment scores [B, num_structures]
- `position_priors`: Learned anatomical position priors

### 3. AbsenceDetectionHead

Determines structure presence/absence using global features and alignment scores.

**Parameters:**
- `num_structures`: Number of anatomical structures
- `feature_dim`: Dimension of input features

**Components:**
- Global adaptive pooling for feature aggregation
- Per-structure MLP classifiers for binary presence/absence classification
- Confidence estimation head

**Outputs:**
- `absence_logits`: Raw logits for presence/absence [B, num_structures, 2]
- `presence_probs`: Probability of structure presence [B, num_structures]
- `confidence_scores`: Prediction confidence scores [B, num_structures]

### 4. SegmentationHead

Produces structure-specific segmentation masks using attention-modulated features.

**Parameters:**
- `in_channels`: Number of input feature channels
- `num_structures`: Number of structures to segment

**Components:**
- Feature refinement network
- Attention-modulated feature processing
- Structure-specific segmentation layers

**Outputs:**
- `segmentation_logits`: Raw segmentation logits [B, num_structures, D, H, W]
- `segmentation_probs`: Segmentation probability maps [B, num_structures, D, H, W]

## Loss Function: SpatialAttentionLoss

A comprehensive multi-component loss function designed for medical segmentation with absence detection.

**Loss Components:**

1. **Segmentation Loss**: BCE loss for present structures only
2. **Dice Loss**: Dice coefficient loss for better boundary detection
3. **Absence Detection Loss**: BCE loss for presence/absence classification
4. **False Positive Suppression**: Penalizes positive predictions for absent structures
5. **Attention Supervision**: Aligns attention maps with ground truth masks
6. **Confidence Regularization**: Penalizes overconfident incorrect predictions

**Parameters:**
- `structure_names`: List of anatomical structure names
- `weights`: Dictionary of loss component weights
- `focal_alpha`, `focal_gamma`: Parameters for focal loss (future use)

**Default Weights:**
```python
{
    'segmentation': 1.0,
    'absence': 2.0,
    'attention_supervision': 0.3,
    'confidence': 0.1,
    'dice': 1.0,
    'false_positive_suppression': 0.5
}
```

## Input/Output Specifications

### Model Inputs

**Primary Input:**
- `x`: Medical image tensor [B, 1, D, H, W]
  - B: Batch size
  - 1: Single channel (grayscale medical images)
  - D, H, W: Spatial dimensions (depth, height, width)

**Optional Parameters:**
- `presence_threshold`: Threshold for presence detection (default: 0.5)

### Model Outputs

The model returns a dictionary containing:

- `attention_maps`: [B, num_structures, D, H, W] - Structure-specific attention maps
- `feature_responses`: [B, num_structures, D, H, W] - Feature responses per structure
- `alignment_scores`: [B, num_structures] - Attention-feature alignment scores
- `position_priors`: [B, num_structures, D, H, W] - Learned anatomical position priors
- `absence_logits`: [B, num_structures, 2] - Raw presence/absence logits
- `presence_probs`: [B, num_structures] - Structure presence probabilities
- `confidence_scores`: [B, num_structures] - Prediction confidence scores
- `segmentation_logits`: [B, num_structures, D, H, W] - Raw segmentation logits
- `segmentation_probs`: [B, num_structures, D, H, W] - Segmentation probability maps

### Training Targets

Training requires targets dictionary with:

- `segmentation_targets`: [B, num_structures, D, H, W] - Binary segmentation masks
- `presence_targets`: [B, num_structures] - Binary presence labels
- `attention_targets`: [B, num_structures, D, H, W] - Optional attention supervision

## Model Instantiation Example

```python
import torch
from your_module import SpatialAttentionMedicalSegmenter, SpatialAttentionLoss

# Define anatomical structures to segment
structure_names = ['left_kidney', 'right_kidney', 'liver', 'spleen', 'pancreas']
num_structures = len(structure_names)

# Model configuration
config = {
    'in_channels': 1,                    # Grayscale medical images
    'num_structures': num_structures,     # Number of anatomical structures
    'spatial_dims': (64, 64, 64)        # Expected input dimensions (D, H, W)
}

# Initialize the model
model = SpatialAttentionMedicalSegmenter(**config)

# Initialize the loss function
loss_weights = {
    'segmentation': 1.0,
    'absence': 2.0,
    'attention_supervision': 0.3,
    'confidence': 0.1,
    'dice': 1.0,
    'false_positive_suppression': 0.5
}

loss_fn = SpatialAttentionLoss(
    structure_names=structure_names,
    weights=loss_weights,
    focal_alpha=0.25,
    focal_gamma=2.0
)

# Model summary
print(f"Model initialized for {num_structures} structures: {structure_names}")
print(f"Expected input shape: [batch_size, 1, {config['spatial_dims'][0]}, {config['spatial_dims'][1]}, {config['spatial_dims'][2]}]")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Example forward pass
model.eval()
batch_size = 2
sample_input = torch.randn(batch_size, 1, *config['spatial_dims'])

with torch.no_grad():
    outputs = model(sample_input)
    print(f"\nForward pass successful!")
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Segmentation output shape: {outputs['segmentation_probs'].shape}")
    print(f"Presence probabilities shape: {outputs['presence_probs'].shape}")
```

## Key Design Considerations

### Handling Missing Structures
The model is specifically designed to handle medical scans where certain anatomical structures may be absent due to:
- Surgical removal
- Congenital absence
- Imaging field of view limitations
- Pathological conditions

### Attention Mechanism
The anatomical attention module incorporates learned position priors to focus on expected anatomical locations while remaining flexible enough to handle anatomical variations.

### Multi-Task Learning
The model simultaneously performs:
1. Binary presence/absence detection per structure
2. Pixel-wise segmentation for present structures
3. Confidence estimation for predictions

### Loss Function Design
The comprehensive loss function ensures:
- Accurate segmentation of present structures
- Correct identification of absent structures
- Prevention of false positive segmentations
- Attention mechanism supervision
- Confidence calibration

## Hardware Requirements

**Memory Scaling:**
Memory usage scales approximately with batch_size × spatial_volume × num_structures. For higher resolution inputs, consider:
- Reducing batch size
- Using gradient checkpointing
- Implementing patch-based inference

## torchinfo summary
```python
================================================================================
SPATIAL ATTENTION MEDICAL SEGMENTATION MODEL ARCHITECTURE
================================================================================
Structures: ['left_kidney', 'right_kidney', 'liver', 'spleen', 'pancreas']
Input shape: (2, 1, 64, 64, 64)
================================================================================
======================================================================================================================================================================
Layer (type (var_name))                                                Input Shape      Output Shape     Param #          Param %          Kernel Shape     Mult-Adds
======================================================================================================================================================================
SpatialAttentionMedicalSegmenter (SpatialAttentionMedicalSegmenter)    [2, 1, 64, 64, 64] [2, 5, 64, 64, 64] --                    --          --               --
├─Encoder (encoder)                                                    [2, 1, 64, 64, 64] [2, 128, 64, 64, 64] --                    --          --               --
│    └─Sequential (enc1)                                               [2, 1, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 1, 64, 64, 64] [2, 16, 64, 64, 64] 448                0.01%          [3, 3, 3]        234,881,024
│    │    └─BatchNorm3d (1)                                            [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 32                 0.00%          --               64
│    │    └─ReLU (2)                                                   [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 6,928              0.09%          [3, 3, 3]        3,632,267,264
│    │    └─BatchNorm3d (4)                                            [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 32                 0.00%          --               64
│    │    └─ReLU (5)                                                   [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    └─MaxPool3d (pool)                                                [2, 16, 64, 64, 64] [2, 16, 32, 32, 32] --                    --          2                --
│    └─Sequential (enc2)                                               [2, 16, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 16, 32, 32, 32] [2, 32, 32, 32, 32] 13,856             0.17%          [3, 3, 3]        908,066,816
│    │    └─BatchNorm3d (1)                                            [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 64                 0.00%          --               128
│    │    └─ReLU (2)                                                   [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 27,680             0.35%          [3, 3, 3]        1,814,036,480
│    │    └─BatchNorm3d (4)                                            [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 64                 0.00%          --               128
│    │    └─ReLU (5)                                                   [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    └─MaxPool3d (pool)                                                [2, 32, 32, 32, 32] [2, 32, 16, 16, 16] --               (recursive)      2                --
│    └─Sequential (enc3)                                               [2, 32, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 32, 16, 16, 16] [2, 64, 16, 16, 16] 55,360             0.69%          [3, 3, 3]        453,509,120
│    │    └─BatchNorm3d (1)                                            [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 128                0.00%          --               256
│    │    └─ReLU (2)                                                   [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 110,656            1.38%          [3, 3, 3]        906,493,952
│    │    └─BatchNorm3d (4)                                            [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 128                0.00%          --               256
│    │    └─ReLU (5)                                                   [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    └─MaxPool3d (pool)                                                [2, 64, 16, 16, 16] [2, 64, 8, 8, 8] --               (recursive)      2                --
│    └─Sequential (enc4)                                               [2, 64, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 64, 8, 8, 8] [2, 128, 8, 8, 8] 221,312            2.76%          [3, 3, 3]        226,623,488
│    │    └─BatchNorm3d (1)                                            [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 256                0.00%          --               512
│    │    └─ReLU (2)                                                   [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 442,496            5.52%          [3, 3, 3]        453,115,904
│    │    └─BatchNorm3d (4)                                            [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 256                0.00%          --               512
│    │    └─ReLU (5)                                                   [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    └─MaxPool3d (pool)                                                [2, 128, 8, 8, 8] [2, 128, 4, 4, 4] --               (recursive)      2                --
│    └─Sequential (enc5)                                               [2, 128, 4, 4, 4] [2, 256, 4, 4, 4] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 128, 4, 4, 4] [2, 256, 4, 4, 4] 884,992           11.04%          [3, 3, 3]        113,278,976
│    │    └─BatchNorm3d (1)                                            [2, 256, 4, 4, 4] [2, 256, 4, 4, 4] 512                0.01%          --               1,024
│    │    └─ReLU (2)                                                   [2, 256, 4, 4, 4] [2, 256, 4, 4, 4] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 256, 4, 4, 4] [2, 256, 4, 4, 4] 1,769,728         22.07%          [3, 3, 3]        226,525,184
│    │    └─BatchNorm3d (4)                                            [2, 256, 4, 4, 4] [2, 256, 4, 4, 4] 512                0.01%          --               1,024
│    │    └─ReLU (5)                                                   [2, 256, 4, 4, 4] [2, 256, 4, 4, 4] --                    --          --               --
│    └─ConvTranspose3d (up4)                                           [2, 256, 4, 4, 4] [2, 128, 8, 8, 8] 262,272            3.27%          [2, 2, 2]        268,566,528
│    └─Sequential (dec4)                                               [2, 256, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 256, 8, 8, 8] [2, 128, 8, 8, 8] 884,864           11.03%          [3, 3, 3]        906,100,736
│    │    └─BatchNorm3d (1)                                            [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 256                0.00%          --               512
│    │    └─ReLU (2)                                                   [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 442,496            5.52%          [3, 3, 3]        453,115,904
│    │    └─BatchNorm3d (4)                                            [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] 256                0.00%          --               512
│    │    └─ReLU (5)                                                   [2, 128, 8, 8, 8] [2, 128, 8, 8, 8] --                    --          --               --
│    └─ConvTranspose3d (up3)                                           [2, 128, 8, 8, 8] [2, 64, 16, 16, 16] 65,600             0.82%          [2, 2, 2]        537,395,200
│    └─Sequential (dec3)                                               [2, 128, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 128, 16, 16, 16] [2, 64, 16, 16, 16] 221,248            2.76%          [3, 3, 3]        1,812,463,616
│    │    └─BatchNorm3d (1)                                            [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 128                0.00%          --               256
│    │    └─ReLU (2)                                                   [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 110,656            1.38%          [3, 3, 3]        906,493,952
│    │    └─BatchNorm3d (4)                                            [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] 128                0.00%          --               256
│    │    └─ReLU (5)                                                   [2, 64, 16, 16, 16] [2, 64, 16, 16, 16] --                    --          --               --
│    └─ConvTranspose3d (up2)                                           [2, 64, 16, 16, 16] [2, 32, 32, 32, 32] 16,416             0.20%          [2, 2, 2]        1,075,838,976
│    └─Sequential (dec2)                                               [2, 64, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 64, 32, 32, 32] [2, 32, 32, 32, 32] 55,328             0.69%          [3, 3, 3]        3,625,975,808
│    │    └─BatchNorm3d (1)                                            [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 64                 0.00%          --               128
│    │    └─ReLU (2)                                                   [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 27,680             0.35%          [3, 3, 3]        1,814,036,480
│    │    └─BatchNorm3d (4)                                            [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] 64                 0.00%          --               128
│    │    └─ReLU (5)                                                   [2, 32, 32, 32, 32] [2, 32, 32, 32, 32] --                    --          --               --
│    └─ConvTranspose3d (up1)                                           [2, 32, 32, 32, 32] [2, 16, 64, 64, 64] 4,112              0.05%          [2, 2, 2]        2,155,872,256
│    └─Sequential (dec1)                                               [2, 32, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 32, 64, 64, 64] [2, 16, 64, 64, 64] 13,840             0.17%          [3, 3, 3]        7,256,145,920
│    │    └─BatchNorm3d (1)                                            [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 32                 0.00%          --               64
│    │    └─ReLU (2)                                                   [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 6,928              0.09%          [3, 3, 3]        3,632,267,264
│    │    └─BatchNorm3d (4)                                            [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] 32                 0.00%          --               64
│    │    └─ReLU (5)                                                   [2, 16, 64, 64, 64] [2, 16, 64, 64, 64] --                    --          --               --
│    └─Conv3d (final_conv)                                             [2, 16, 64, 64, 64] [2, 128, 64, 64, 64] 2,176              0.03%          [1, 1, 1]        1,140,850,688
├─AnatomicalAttentionModule (attention_module)                         [2, 128, 64, 64, 64] [2, 5, 64, 64, 64] 1,310,720         16.34%          --               --
│    └─Sequential (attention_conv)                                     [2, 128, 64, 64, 64] [2, 5, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 128, 64, 64, 64] [2, 64, 64, 64, 64] 221,248            2.76%          [3, 3, 3]        115,997,671,424
│    │    └─BatchNorm3d (1)                                            [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] 128                0.00%          --               256
│    │    └─ReLU (2)                                                   [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 64, 64, 64, 64] [2, 5, 64, 64, 64] 325                0.00%          [1, 1, 1]        170,393,600
│    │    └─Sigmoid (4)                                                [2, 5, 64, 64, 64] [2, 5, 64, 64, 64] --                    --          --               --
│    └─Sequential (feature_matcher)                                    [2, 128, 64, 64, 64] [2, 5, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 128, 64, 64, 64] [2, 128, 64, 64, 64] 442,496            5.52%          [3, 3, 3]        231,995,342,848
│    │    └─BatchNorm3d (1)                                            [2, 128, 64, 64, 64] [2, 128, 64, 64, 64] 256                0.00%          --               512
│    │    └─ReLU (2)                                                   [2, 128, 64, 64, 64] [2, 128, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 128, 64, 64, 64] [2, 5, 64, 64, 64] 645                0.01%          [1, 1, 1]        338,165,760
├─AbsenceDetectionHead (absence_detector)                              [2, 128, 64, 64, 64] [2, 5]           --                    --          --               --
│    └─AdaptiveAvgPool3d (global_pool)                                 [2, 128, 64, 64, 64] [2, 128, 1, 1, 1] --                    --          --               --
│    └─ModuleList (absence_classifiers)                                --               --               --                    --          --               --
│    │    └─Sequential (0)                                             [2, 129]         [2, 2]           10,466             0.13%          --               20,932
│    │    └─Sequential (1)                                             [2, 129]         [2, 2]           10,466             0.13%          --               20,932
│    │    └─Sequential (2)                                             [2, 129]         [2, 2]           10,466             0.13%          --               20,932
│    │    └─Sequential (3)                                             [2, 129]         [2, 2]           10,466             0.13%          --               20,932
│    │    └─Sequential (4)                                             [2, 129]         [2, 2]           10,466             0.13%          --               20,932
│    └─Sequential (confidence_head)                                    [2, 128]         [2, 5]           --                    --          --               --
│    │    └─Linear (0)                                                 [2, 128]         [2, 64]          8,256              0.10%          --               16,512
│    │    └─ReLU (1)                                                   [2, 64]          [2, 64]          --                    --          --               --
│    │    └─Linear (2)                                                 [2, 64]          [2, 5]           325                0.00%          --               650
│    │    └─Sigmoid (3)                                                [2, 5]           [2, 5]           --                    --          --               --
├─SegmentationHead (segmentation_head)                                 [2, 128, 64, 64, 64] [2, 5, 64, 64, 64] --                    --          --               --
│    └─Sequential (feature_refine)                                     [2, 128, 64, 64, 64] [2, 64, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (0)                                                 [2, 128, 64, 64, 64] [2, 64, 64, 64, 64] 221,248            2.76%          [3, 3, 3]        115,997,671,424
│    │    └─BatchNorm3d (1)                                            [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] 128                0.00%          --               256
│    │    └─ReLU (2)                                                   [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] --                    --          --               --
│    │    └─Conv3d (3)                                                 [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] 110,656            1.38%          [3, 3, 3]        58,015,612,928
│    │    └─BatchNorm3d (4)                                            [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] 128                0.00%          --               256
│    │    └─ReLU (5)                                                   [2, 64, 64, 64, 64] [2, 64, 64, 64, 64] --                    --          --               --
│    └─Conv3d (segmentation_layer)                                     [10, 64, 64, 64, 64] [10, 5, 64, 64, 64] 325                0.00%          [1, 1, 1]        851,968,000
======================================================================================================================================================================
Total params: 8,019,230
Trainable params: 8,019,230
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 557.92
======================================================================================================================================================================
Input size (MB): 2.10
Forward/backward pass size (MB): 4171.24
Params size (MB): 26.83
Estimated Total Size (MB): 4200.18
======================================================================================================================================================================

================================================================================
MODEL STATISTICS
================================================================================
Total Parameters: 8,019,230
Trainable Parameters: 8,019,230
Non-trainable Parameters: 0
Model Size: 30.60 MB
Estimated Input Memory: 2.00 MB
Estimated Total Memory (forward pass): ~36.6 MB
```
