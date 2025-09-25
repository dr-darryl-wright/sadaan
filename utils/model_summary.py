#!/usr/bin/env python3
"""
Minimal script to display the Spatial Attention Medical Segmentation Model architecture
using torchinfo.
"""

import torch
from torchinfo import summary

# Import the model (adjust import path as needed)
from your_model_file import SpatialAttentionMedicalSegmenter


def print_model_summary():
    """Print detailed model architecture summary using torchinfo"""

    # Model configuration
    structure_names = ['left_kidney', 'right_kidney', 'liver', 'spleen', 'pancreas']
    num_structures = len(structure_names)
    spatial_dims = (64, 64, 64)

    # Initialize model
    model = SpatialAttentionMedicalSegmenter(
        in_channels=1,
        num_structures=num_structures,
        spatial_dims=spatial_dims
    )

    # Input shape: [batch_size, channels, depth, height, width]
    input_size = (2, 1, *spatial_dims)  # batch_size=2 for demonstration

    print("=" * 80)
    print("SPATIAL ATTENTION MEDICAL SEGMENTATION MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Structures: {structure_names}")
    print(f"Input shape: {input_size}")
    print("=" * 80)

    # Generate detailed summary
    model_summary = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"],
        col_width=16,
        row_settings=["var_names"],
        depth=3,  # Show 3 levels of nested modules
        device="cpu",
        dtypes=[torch.float32],
        verbose=1
    )

    print("\n" + "=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

    # Estimate model size
    param_size = total_params * 4  # 4 bytes per float32 parameter
    buffer_size = sum([buf.numel() * buf.element_size() for buf in model.buffers()])
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    print(f"Model Size: {model_size_mb:.2f} MB")

    # Memory estimation for forward pass (rough estimate)
    input_size_mb = (2 * 1 * 64 * 64 * 64 * 4) / 1024 / 1024  # input tensor size
    print(f"Estimated Input Memory: {input_size_mb:.2f} MB")
    print(f"Estimated Total Memory (forward pass): ~{model_size_mb + input_size_mb * 3:.1f} MB")


if __name__ == "__main__":
    print_model_summary()
