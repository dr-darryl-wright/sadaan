#!/usr/bin/env python3
"""
Simple script to validate synthetic medical data masks
Tests for common issues that cause low Dice scores
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def validate_synthetic_masks(dataset_path):
    """
    Validate synthetic dataset masks for common issues

    Args:
        dataset_path: Path to synthetic_medical_dataset directory
    """

    dataset_path = Path(dataset_path)

    print("=" * 60)
    print("SYNTHETIC DATA MASK VALIDATION")
    print("=" * 60)

    # Load metadata
    metadata_path = dataset_path / 'metadata.json'
    if not metadata_path.exists():
        print(f"❌ ERROR: metadata.json not found at {metadata_path}")
        return False

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    structure_names = metadata['structure_names']
    image_size = metadata['image_size']

    print(f"Dataset: {dataset_path}")
    print(f"Structures: {structure_names}")
    print(f"Image size: {image_size}")
    print()

    # Check each split
    issues_found = []

    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"⚠️  Split {split} not found, skipping...")
            continue

        print(f"Checking {split} split...")

        # Load data
        data_path = split_path / 'data.npz'
        if not data_path.exists():
            print(f"❌ ERROR: {split}/data.npz not found")
            issues_found.append(f"{split}: Missing data.npz")
            continue

        data = np.load(data_path)

        images = data['images']
        masks = data['masks']
        presence_labels = data['presence_labels']

        print(f"  Samples: {len(images)}")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Presence labels shape: {presence_labels.shape}")

        # Test 1: Check data types and ranges
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Mask range: [{masks.min():.2f}, {masks.max():.2f}]")
        print(f"  Presence range: [{presence_labels.min()}, {presence_labels.max()}]")

        if masks.min() < 0 or masks.max() > 1:
            issues_found.append(f"{split}: Masks not in [0,1] range")
            print("    ❌ ERROR: Masks should be in [0,1] range")

        if not np.all(np.isin(presence_labels, [0, 1])):
            issues_found.append(f"{split}: Presence labels not binary")
            print("    ❌ ERROR: Presence labels should be 0 or 1")

        # Test 2: Check mask-presence consistency
        print(f"  Checking mask-presence consistency...")
        consistency_errors = 0

        for sample_idx in range(min(10, len(images))):  # Check first 10 samples
            for struct_idx in range(len(structure_names)):
                mask = masks[sample_idx, struct_idx]
                presence = presence_labels[sample_idx, struct_idx]

                has_pixels = np.any(mask > 0.5)

                if presence == 1 and not has_pixels:
                    consistency_errors += 1
                    if consistency_errors <= 3:  # Only print first few
                        print(f"    ❌ Sample {sample_idx}, {structure_names[struct_idx]}: "
                              f"Present=1 but no mask pixels")

                if presence == 0 and has_pixels:
                    consistency_errors += 1
                    if consistency_errors <= 3:
                        print(f"    ❌ Sample {sample_idx}, {structure_names[struct_idx]}: "
                              f"Present=0 but has mask pixels")

        if consistency_errors > 0:
            issues_found.append(f"{split}: {consistency_errors} mask-presence mismatches")
            print(f"    Total consistency errors: {consistency_errors}")
        else:
            print("    ✅ Mask-presence consistency OK")

        # Test 3: Check mask statistics
        print(f"  Mask statistics:")
        positive_ratios = []

        for struct_idx, struct_name in enumerate(structure_names):
            struct_masks = masks[:, struct_idx]
            present_samples = presence_labels[:, struct_idx] == 1

            if np.any(present_samples):
                present_masks = struct_masks[present_samples]
                positive_ratio = np.mean(present_masks > 0.5)
                positive_ratios.append(positive_ratio)
                print(f"    {struct_name}: {positive_ratio:.4f} positive pixels (when present)")

                if positive_ratio < 0.001:
                    issues_found.append(f"{split}: {struct_name} has very few positive pixels")
                    print(f"      ⚠️  Very sparse masks detected")
                elif positive_ratio > 0.5:
                    issues_found.append(f"{split}: {struct_name} has too many positive pixels")
                    print(f"      ⚠️  Very dense masks detected")
            else:
                print(f"    {struct_name}: Never present in this split")

        # Test 4: Visualize a few samples
        print(f"  Creating sample visualizations...")

        # Find a sample with at least one present structure
        sample_idx = 0
        for i in range(len(presence_labels)):
            if np.any(presence_labels[i] == 1):
                sample_idx = i
                break

        fig, axes = plt.subplots(2, len(structure_names), figsize=(3 * len(structure_names), 6))
        if len(structure_names) == 1:
            axes = axes.reshape(2, 1)

        # Get middle slice
        slice_idx = images.shape[-1] // 2
        img_slice = images[sample_idx, :, :, slice_idx]

        for struct_idx, struct_name in enumerate(structure_names):
            mask_slice = masks[sample_idx, struct_idx, :, :, slice_idx]
            present = presence_labels[sample_idx, struct_idx]

            # Original image
            axes[0, struct_idx].imshow(img_slice, cmap='gray')
            axes[0, struct_idx].set_title(f'{struct_name}\nPresent: {present}')
            axes[0, struct_idx].axis('off')

            # Mask overlay
            axes[1, struct_idx].imshow(img_slice, cmap='gray', alpha=0.7)
            if present == 1:
                axes[1, struct_idx].imshow(mask_slice, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[1, struct_idx].set_title(f'Mask\nPos pixels: {np.mean(mask_slice > 0.5):.3f}')
            axes[1, struct_idx].axis('off')

        plt.suptitle(f'{split} Split - Sample {sample_idx} (Slice {slice_idx})')
        plt.tight_layout()
        plt.savefig(f'mask_validation_{split}.png', dpi=100, bbox_inches='tight')
        plt.show()

        print(f"  Saved visualization: mask_validation_{split}.png")
        print()

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if len(issues_found) == 0:
        print("✅ All tests passed! Masks appear correctly configured.")
        return True
    else:
        print(f"❌ Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"  - {issue}")

        print("\nCOMMON FIXES:")
        print("1. Ensure masks are in [0,1] range, not {0,255}")
        print("2. Check that presence_labels match actual mask content")
        print("3. Verify mask generation code creates reasonable positive pixel ratios")
        print("4. Make sure mask dimensions match image dimensions")

        return False


def quick_mask_test(dataset_path, num_samples=5):
    """
    Quick test of just a few samples for rapid debugging
    """

    print("QUICK MASK TEST")
    print("-" * 30)

    # Load just validation data
    val_data_path = Path(dataset_path) / 'val' / 'data.npz'

    if not val_data_path.exists():
        print(f"❌ Cannot find {val_data_path}")
        return

    data = np.load(val_data_path)

    images = data['images']
    masks = data['masks']
    presence_labels = data['presence_labels']

    print(f"Testing {num_samples} samples...")

    for i in range(min(num_samples, len(images))):
        print(f"\nSample {i}:")

        for struct_idx in range(masks.shape[1]):
            mask = masks[i, struct_idx]
            presence = presence_labels[i, struct_idx]

            pos_pixels = np.mean(mask > 0.5)
            max_val = mask.max()
            min_val = mask.min()

            status = "✅" if (presence == 1 and pos_pixels > 0) or (presence == 0 and pos_pixels == 0) else "❌"

            print(f"  Struct {struct_idx}: Present={presence}, "
                  f"PosPix={pos_pixels:.4f}, Range=[{min_val:.3f},{max_val:.3f}] {status}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_masks.py <dataset_path>")
        print("Example: python validate_masks.py ./synthetic_medical_dataset")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Run quick test first
    quick_mask_test(dataset_path)

    print("\n" + "=" * 60)

    # Run full validation
    success = validate_synthetic_masks(dataset_path)

    if success:
        print("\n🎉 Dataset validation completed successfully!")
    else:
        print("\n⚠️  Issues found. Fix these before training.")