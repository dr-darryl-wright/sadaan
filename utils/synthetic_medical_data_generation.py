import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
import h5py
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
from pathlib import Path



@dataclass
class StructureTemplate:
    """Template for anatomical structure"""
    center: Tuple[float, float, float]  # Relative coordinates (0-1)
    size: Tuple[float, float, float]  # Relative size (0-1)
    intensity: float  # Base intensity value
    shape: str = 'ellipsoid'  # Shape type


class SyntheticAnatomyGenerator:
    """Generate synthetic anatomical data for test medical AI models"""

    def __init__(self, image_size: Tuple[int, int, int] = (128, 128, 64)):
        self.image_size = image_size
        self.anatomy_templates = self._create_anatomy_templates()

    def _create_anatomy_templates(self) -> Dict[str, StructureTemplate]:
        """Define anatomical structure templates based on realistic anatomy"""

        templates = {
            # Thoracic structures
            'left_lung': StructureTemplate(
                center=(0.25, 0.3, 0.5),
                size=(0.15, 0.25, 0.35),
                intensity=150,
                shape='ellipsoid'
            ),
            'right_lung': StructureTemplate(
                center=(0.75, 0.3, 0.5),
                size=(0.18, 0.25, 0.35),  # Right lung slightly larger
                intensity=150,
                shape='ellipsoid'
            ),
            'heart': StructureTemplate(
                center=(0.45, 0.35, 0.5),
                size=(0.12, 0.15, 0.2),
                intensity=200,
                shape='ellipsoid'
            ),

            # Abdominal structures
            'liver': StructureTemplate(
                center=(0.65, 0.6, 0.5),
                size=(0.25, 0.2, 0.3),
                intensity=180,
                shape='ellipsoid'
            ),
            'spleen': StructureTemplate(
                center=(0.25, 0.65, 0.5),
                size=(0.08, 0.1, 0.15),
                intensity=170,
                shape='ellipsoid'
            ),
            'pancreas': StructureTemplate(
                center=(0.45, 0.65, 0.45),
                size=(0.15, 0.04, 0.08),
                intensity=160,
                shape='ellipsoid'
            ),

            # Kidneys
            'left_kidney': StructureTemplate(
                center=(0.3, 0.7, 0.4),
                size=(0.08, 0.12, 0.15),
                intensity=160,
                shape='kidney'  # Special kidney shape
            ),
            'right_kidney': StructureTemplate(
                center=(0.7, 0.7, 0.4),
                size=(0.08, 0.12, 0.15),
                intensity=160,
                shape='kidney'
            ),

            # Vertebrae (simplified as small cylinders)
            'L1': StructureTemplate(
                center=(0.5, 0.85, 0.3),
                size=(0.06, 0.06, 0.08),
                intensity=220,
                shape='cylinder'
            ),
            'L2': StructureTemplate(
                center=(0.5, 0.85, 0.4),
                size=(0.06, 0.06, 0.08),
                intensity=220,
                shape='cylinder'
            ),
            'L3': StructureTemplate(
                center=(0.5, 0.85, 0.5),
                size=(0.06, 0.06, 0.08),
                intensity=220,
                shape='cylinder'
            ),
            'L4': StructureTemplate(
                center=(0.5, 0.85, 0.6),
                size=(0.06, 0.06, 0.08),
                intensity=220,
                shape='cylinder'
            ),
            'L5': StructureTemplate(
                center=(0.5, 0.85, 0.7),
                size=(0.06, 0.06, 0.08),
                intensity=220,
                shape='cylinder'
            ),
        }

        return templates

    def create_ellipsoid_mask(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Create ellipsoid-shaped mask"""
        z, y, x = np.ogrid[:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        # Ellipsoid equation: ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² <= 1
        mask = (((z - center[0]) / max(size[0], 1)) ** 2 +
                ((y - center[1]) / max(size[1], 1)) ** 2 +
                ((x - center[2]) / max(size[2], 1)) ** 2) <= 1

        return mask

    def create_kidney_mask(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Create kidney-shaped mask (bean-like ellipsoid with indentation)"""
        base_mask = self.create_ellipsoid_mask(center, size)

        # Create indentation for renal hilum
        indent_center = center.copy()
        indent_center[1] -= size[1] * 0.3  # Indent towards center
        indent_size = size * 0.4
        indent_mask = self.create_ellipsoid_mask(indent_center, indent_size)

        # Subtract indentation
        kidney_mask = base_mask & ~indent_mask

        return kidney_mask

    def create_cylinder_mask(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Create cylindrical mask (for vertebrae)"""
        z, y, x = np.ogrid[:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        # Cylinder along z-axis
        radial_dist = np.sqrt(((y - center[1]) / max(size[1], 1)) ** 2 +
                              ((x - center[2]) / max(size[2], 1)) ** 2)
        height_condition = np.abs(z - center[0]) <= size[0]

        mask = (radial_dist <= 1) & height_condition

        return mask

    def create_structure_mask(self, template: StructureTemplate, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Create mask based on structure shape type"""
        if template.shape == 'ellipsoid':
            return self.create_ellipsoid_mask(center, size)
        elif template.shape == 'kidney':
            return self.create_kidney_mask(center, size)
        elif template.shape == 'cylinder':
            return self.create_cylinder_mask(center, size)
        else:
            return self.create_ellipsoid_mask(center, size)

    def create_bias_field(self) -> np.ndarray:
        """Create smooth intensity bias field"""
        # Create smooth random field at lower resolution
        low_res_shape = tuple(s // 8 for s in self.image_size)
        small_field = np.random.normal(1.0, 0.15, low_res_shape)

        # Upsample to full resolution
        bias_field = ndimage.zoom(small_field, 8, order=3)

        # Ensure exact shape match
        bias_field = bias_field[:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        # Smooth further
        bias_field = ndimage.gaussian_filter(bias_field, sigma=2.0)

        return bias_field

    def generate_individual_structure(self, struct_name: str, template: StructureTemplate,
                                      center: np.ndarray, size: np.ndarray, intensity: float) -> np.ndarray:
        """Generate a single structure on blank background with all processing applied"""

        # Create structure on blank background
        structure_image = np.zeros(self.image_size, dtype=np.float32)

        # Convert relative coordinates to absolute
        abs_center = (center * np.array(self.image_size)).astype(int)
        abs_size = (size * np.array(self.image_size) / 2).astype(int)  # Radius, not diameter

        # Create initial structure mask
        initial_mask = self.create_structure_mask(template, abs_center, abs_size)

        if not initial_mask.any():
            return structure_image

        # Add structure with internal intensity variation
        internal_variation = np.random.normal(0, 15, initial_mask.sum())
        structure_intensities = intensity + internal_variation
        structure_intensities = np.clip(structure_intensities, intensity * 0.7, intensity * 1.3)
        structure_image[initial_mask] = structure_intensities

        # Apply imaging artifacts to individual structure
        # 1. Gaussian blur (imaging resolution)
        structure_image = ndimage.gaussian_filter(structure_image, sigma=0.8)

        # 2. Intensity bias field (common in MRI/CT)
        bias_field = self.create_bias_field()
        structure_image *= bias_field

        # 3. Clip to valid intensity range (but don't add noise yet)
        structure_image = np.clip(structure_image, 0, 255)

        return structure_image

    def generate_sample(self, present_structures: List[str],
                        position_noise: float = 0.05,
                        size_noise: float = 0.2,
                        intensity_noise: float = 0.15) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
        """Generate one synthetic medical image sample"""

        # Initialize final image with zeros (background will be added later)
        final_image = np.zeros(self.image_size, dtype=np.float32)

        # Storage for masks and presence labels
        masks = {}
        presence_labels = {}

        # Initialize all structures as absent
        for struct_name in self.anatomy_templates.keys():
            presence_labels[struct_name] = 0
            masks[struct_name] = np.zeros(self.image_size, dtype=bool)

        # Generate each present structure individually
        for struct_name in present_structures:
            if struct_name not in self.anatomy_templates:
                continue

            template = self.anatomy_templates[struct_name]
            presence_labels[struct_name] = 1

            # Add random variations
            center = np.array(template.center)
            noise = np.random.normal(0, position_noise, 3)
            center += noise
            center = np.clip(center, 0.1, 0.9)  # Keep within image bounds

            size = np.array(template.size)
            size_variation = 1 + np.random.normal(0, size_noise, 3)
            size *= size_variation
            size = np.clip(size, 0.02, 0.5)  # Reasonable size limits

            intensity = template.intensity
            intensity *= (1 + np.random.normal(0, intensity_noise))
            intensity = np.clip(intensity, 50, 250)

            # Generate individual structure with all processing
            structure_image = self.generate_individual_structure(
                struct_name, template, center, size, intensity
            )

            # Generate mask from processed structure (threshold at > 10)
            masks[struct_name] = structure_image > 10

            # Add structure to final image
            final_image += structure_image

        # Now add background tissue noise to the final composite image
        background_noise = np.random.normal(40, 20, self.image_size)
        background_noise = np.clip(background_noise, 0, 80)
        final_image += background_noise

        # Add final random noise
        final_noise = np.random.normal(0, 8, self.image_size)
        final_image += final_noise

        # Final clipping
        final_image = np.clip(final_image, 0, 255)

        return final_image, masks, presence_labels


class SyntheticDatasetGenerator:
    """Generate complete datasets for training and testing with HDF5 support"""

    def __init__(self, image_size: Tuple[int, int, int] = (128, 128, 64)):
        self.generator = SyntheticAnatomyGenerator(image_size)
        self.scenarios = self._create_test_scenarios()

    def _create_test_scenarios(self) -> List[Dict]:
        """Define test scenarios with different absence patterns"""

        all_structures = list(self.generator.anatomy_templates.keys())

        scenarios = [
            # Complete anatomy
            {
                'name': 'normal_complete',
                'present': all_structures.copy(),
                'weight': 0.3  # Higher weight for normal cases
            },
            {
                'name': 'normal_thoracic',
                'present': ['left_lung', 'right_lung', 'heart'],
                'weight': 0.1
            },
            {
                'name': 'normal_abdominal',
                'present': ['liver', 'spleen', 'pancreas', 'left_kidney', 'right_kidney'],
                'weight': 0.1
            },

            # Single organ missing (surgical cases)
            {
                'name': 'left_nephrectomy',
                'present': [s for s in all_structures if s != 'left_kidney'],
                'weight': 0.08
            },
            {
                'name': 'right_nephrectomy',
                'present': [s for s in all_structures if s != 'right_kidney'],
                'weight': 0.08
            },
            {
                'name': 'splenectomy',
                'present': [s for s in all_structures if s != 'spleen'],
                'weight': 0.06
            },
            {
                'name': 'left_lung_resection',
                'present': [s for s in all_structures if s != 'left_lung'],
                'weight': 0.04
            },

            # Bilateral missing
            {
                'name': 'bilateral_nephrectomy',
                'present': [s for s in all_structures if 'kidney' not in s],
                'weight': 0.03
            },
            {
                'name': 'bilateral_lung_resection',
                'present': [s for s in all_structures if 'lung' not in s],
                'weight': 0.02
            },

            # Vertebral missing (trauma/congenital)
            {
                'name': 'missing_L4',
                'present': [s for s in all_structures if s != 'L4'],
                'weight': 0.05
            },
            {
                'name': 'missing_L2_L3',
                'present': [s for s in all_structures if s not in ['L2', 'L3']],
                'weight': 0.03
            },

            # Complex multi-organ missing
            {
                'name': 'complex_abdominal_surgery',
                'present': [s for s in all_structures if s not in ['spleen', 'pancreas', 'left_kidney']],
                'weight': 0.04
            },
            {
                'name': 'extensive_resection',
                'present': ['heart', 'right_lung', 'liver', 'right_kidney'] + [f'L{i}' for i in range(1, 6)],
                'weight': 0.02
            },

            # Edge cases
            {
                'name': 'minimal_structures',
                'present': ['heart', 'liver', 'L3'],
                'weight': 0.02
            }
        ]

        return scenarios

    def generate_dataset(self,
                         total_samples: int = 1000,
                         validation_split: float = 0.2,
                         test_split: float = 0.1,
                         position_noise: float = 0.05,
                         size_noise: float = 0.2,
                         intensity_noise: float = 0.15) -> Dict:
        """Generate complete dataset with train/val/test splits"""

        # Calculate samples per scenario based on weights
        weights = np.array([s['weight'] for s in self.scenarios])
        weights = weights / weights.sum()
        samples_per_scenario = (weights * total_samples).astype(int)

        # Ensure we generate exactly the requested number of samples
        samples_per_scenario[-1] += total_samples - samples_per_scenario.sum()

        print("Generating dataset...")
        print(f"Total samples: {total_samples}")
        print(f"Scenarios: {len(self.scenarios)}")

        all_images = []
        all_masks = []
        all_presence_labels = []
        all_scenarios = []

        for scenario, n_samples in zip(self.scenarios, samples_per_scenario):
            if n_samples == 0:
                continue

            print(f"  {scenario['name']}: {n_samples} samples")

            for _ in range(n_samples):
                image, masks, presence_labels = self.generator.generate_sample(
                    present_structures=scenario['present'],
                    position_noise=position_noise,
                    size_noise=size_noise,
                    intensity_noise=intensity_noise
                )

                all_images.append(image)
                all_masks.append(masks)
                all_presence_labels.append(presence_labels)
                all_scenarios.append(scenario['name'])

        # Convert to arrays
        images_array = np.stack(all_images)

        # Convert masks to structured format
        structure_names = list(self.generator.anatomy_templates.keys())
        masks_array = np.zeros((len(all_images), len(structure_names), *self.generator.image_size), dtype=bool)
        presence_array = np.zeros((len(all_images), len(structure_names)), dtype=int)

        for i, (masks_dict, presence_dict) in enumerate(zip(all_masks, all_presence_labels)):
            for j, struct_name in enumerate(structure_names):
                masks_array[i, j] = masks_dict[struct_name]
                presence_array[i, j] = presence_dict[struct_name]

        # Create train/val/test splits
        indices = np.arange(len(images_array))
        train_idx, temp_idx = train_test_split(indices, test_size=(validation_split + test_split),
                                               random_state=42, stratify=all_scenarios)

        if test_split > 0:
            val_size = validation_split / (validation_split + test_split)
            val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size),
                                                 random_state=42)
        else:
            val_idx = temp_idx
            test_idx = np.array([])

        dataset = {
            'train': {
                'images': images_array[train_idx],
                'masks': masks_array[train_idx],
                'presence_labels': presence_array[train_idx],
                'scenarios': [all_scenarios[i] for i in train_idx]
            },
            'val': {
                'images': images_array[val_idx],
                'masks': masks_array[val_idx],
                'presence_labels': presence_array[val_idx],
                'scenarios': [all_scenarios[i] for i in val_idx]
            },
            'structure_names': structure_names,
            'image_size': self.generator.image_size,
            'generation_params': {
                'position_noise': position_noise,
                'size_noise': size_noise,
                'intensity_noise': intensity_noise
            }
        }

        if len(test_idx) > 0:
            dataset['test'] = {
                'images': images_array[test_idx],
                'masks': masks_array[test_idx],
                'presence_labels': presence_array[test_idx],
                'scenarios': [all_scenarios[i] for i in test_idx]
            }

        return dataset

    def save_dataset_hdf5(self, dataset: Dict, output_dir: str):
        """Save dataset in HDF5 format for memory-efficient loading"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Saving dataset in HDF5 format...")

        for split in ['train', 'val', 'test']:
            if split not in dataset:
                continue

            print(f"  Saving {split} split...")
            split_file = output_path / f'{split}.h5'

            with h5py.File(split_file, 'w') as f:
                # Create datasets with compression and chunking for efficient access
                # Chunk by individual samples for efficient random access
                chunk_size = (1, *dataset[split]['images'].shape[1:])

                f.create_dataset('images',
                                 data=dataset[split]['images'],
                                 compression='gzip',
                                 compression_opts=6,  # Good balance of compression/speed
                                 chunks=chunk_size)

                # Chunk masks by sample as well
                mask_chunk_size = (1, *dataset[split]['masks'].shape[1:])
                f.create_dataset('masks',
                                 data=dataset[split]['masks'],
                                 compression='gzip',
                                 compression_opts=6,
                                 chunks=mask_chunk_size)

                f.create_dataset('presence_labels',
                                 data=dataset[split]['presence_labels'],
                                 compression='gzip',
                                 compression_opts=6)

                # Store scenarios as strings (HDF5 handles string arrays)
                scenarios_array = np.array(dataset[split]['scenarios'], dtype=h5py.string_dtype())
                f.create_dataset('scenarios', data=scenarios_array)

                # Store metadata as attributes
                f.attrs['n_samples'] = len(dataset[split]['images'])
                f.attrs['image_shape'] = dataset[split]['images'].shape[1:]
                f.attrs['n_structures'] = len(dataset['structure_names'])

        # Save metadata as JSON for easy access
        metadata = {
            'structure_names': dataset['structure_names'],
            'image_size': dataset['image_size'],
            'generation_params': dataset['generation_params'],
            'scenarios': self.scenarios
        }

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"HDF5 dataset saved to {output_path}")

    def save_dataset(self, dataset: Dict, output_dir: str):
        """Save dataset - defaults to HDF5 format"""
        self.save_dataset_hdf5(dataset, output_dir)


class LazyMedicalDataset(Dataset):
    """PyTorch Dataset that loads samples on-demand from HDF5"""

    def __init__(self, data_path: str, split: str = 'train',
                 transform=None, load_masks: bool = True,
                 preload_metadata: bool = True):
        """
        Args:
            data_path: Path to the HDF5 dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform function
            load_masks: Whether to load segmentation masks (set False for classification only)
            preload_metadata: Whether to preload scenarios for faster access
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.load_masks = load_masks

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

        # Optionally preload scenarios for faster access
        if preload_metadata:
            self.scenarios = [s.decode('utf-8') if isinstance(s, bytes) else s
                              for s in self.hdf5_file['scenarios'][:]]
        else:
            self.scenarios = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")

        # Load image - HDF5 efficiently loads only the requested slice
        image = self.hdf5_file['images'][idx]
        presence_labels = self.hdf5_file['presence_labels'][idx]

        # Optionally load masks
        if self.load_masks:
            masks = self.hdf5_file['masks'][idx]
        else:
            masks = None

        # Get scenario if needed
        if self.scenarios is not None:
            scenario = self.scenarios[idx]
        else:
            scenario_bytes = self.hdf5_file['scenarios'][idx]
            scenario = scenario_bytes.decode('utf-8') if isinstance(scenario_bytes, bytes) else scenario_bytes

        # Apply transforms if specified
        if self.transform:
            if masks is not None:
                image, masks = self.transform(image, masks)
            else:
                image = self.transform(image)

        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        presence_labels = torch.from_numpy(presence_labels.astype(np.int64))

        if masks is not None:
            masks = torch.from_numpy(masks.astype(np.bool_))
            return {
                'image': image,
                'masks': masks,
                'presence_labels': presence_labels,
                'scenario': scenario,
                'index': idx
            }
        else:
            return {
                'image': image,
                'presence_labels': presence_labels,
                'scenario': scenario,
                'index': idx
            }

    def __del__(self):
        """Clean up HDF5 file handle"""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()

    def get_structure_names(self):
        """Get list of anatomical structure names"""
        return self.structure_names

    def get_class_distribution(self):
        """Get distribution of presence labels across the dataset"""
        presence_labels = self.hdf5_file['presence_labels'][:]
        presence_counts = presence_labels.sum(axis=0)

        distribution = {}
        for i, struct_name in enumerate(self.structure_names):
            distribution[struct_name] = {
                'present': int(presence_counts[i]),
                'absent': int(self.n_samples - presence_counts[i]),
                'presence_rate': float(presence_counts[i] / self.n_samples)
            }

        return distribution


def create_data_transforms():
    """Create example data transforms"""

    def basic_transform(image, masks=None):
        # Example: Add channel dimension for 2D CNNs or normalize
        # For 3D data, you might want to extract 2D slices

        # Normalize to [0, 1]
        image = image / 255.0

        # Add channel dimension if needed (for 2D processing)
        # image = image[..., None]

        if masks is not None:
            return image, masks
        else:
            return image

    return basic_transform


def visualize_sample(dataset: LazyMedicalDataset, idx: int, slice_idx: Optional[int] = None):
    """Visualize a single sample from the lazy dataset"""
    sample = dataset[idx]

    image = sample['image'].numpy()
    presence_labels = sample['presence_labels'].numpy()
    scenario = sample['scenario']

    if slice_idx is None:
        slice_idx = image.shape[-1] // 2

    # Get middle slice
    img_slice = image[..., slice_idx]
    img_slice = np.rot90(img_slice, -1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Show original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'Original Image\nScenario: {scenario}')
    axes[0].axis('off')

    # Show presence labels as text
    present_structures = [name for name, present in zip(dataset.structure_names, presence_labels) if present]
    absent_structures = [name for name, present in zip(dataset.structure_names, presence_labels) if not present]

    axes[1].text(0.05, 0.95, f"Present ({len(present_structures)}):\n" + "\n".join(present_structures),
                 transform=axes[1].transAxes, fontsize=8, verticalalignment='top', color='green')
    axes[1].text(0.55, 0.95, f"Absent ({len(absent_structures)}):\n" + "\n".join(absent_structures),
                 transform=axes[1].transAxes, fontsize=8, verticalalignment='top', color='red')
    axes[1].set_title('Structure Status')
    axes[1].axis('off')

    # Show individual masks if available
    if 'masks' in sample:
        masks = sample['masks'].numpy()

        plot_idx = 2
        for i, struct_name in enumerate(dataset.structure_names[:6]):
            if plot_idx >= 8:
                break

            mask_slice = masks[i, ..., slice_idx]
            mask_slice = np.rot90(mask_slice, -1)
            masked_img = img_slice.copy()
            masked_img[~mask_slice] = 0

            status = "Present" if presence_labels[i] == 1 else "Absent"
            color = 'green' if presence_labels[i] == 1 else 'red'

            axes[plot_idx].imshow(masked_img, cmap='gray')
            axes[plot_idx].set_title(f'{struct_name}\n({status})', color=color)
            axes[plot_idx].axis('off')
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, 8):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def example_training_loop():
    """Example training loop using the lazy HDF5 dataset"""

    # Create datasets
    train_dataset = LazyMedicalDataset(
        "../data/synthetic_medical_dataset_hdf5",
        split='train',
        load_masks=False,  # Set to True if you need segmentation masks
        transform=create_data_transforms()
    )

    val_dataset = LazyMedicalDataset(
        "../data/synthetic_medical_dataset_hdf5",
        split='val',
        load_masks=False,
        transform=create_data_transforms()
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,
        persistent_workers=True  # Keeps workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print("Dataset Information:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Structures: {train_dataset.get_structure_names()}")

    # Print class distribution
    print("\nClass Distribution (Training):")
    distribution = train_dataset.get_class_distribution()
    for struct_name, stats in distribution.items():
        print(f"  {struct_name}: {stats['presence_rate']:.1%} present ({stats['present']}/{len(train_dataset)})")

    # Example training loop
    print("\nStarting training loop example...")

    for epoch in range(3):  # Just 3 epochs for demo
        print(f"\nEpoch {epoch + 1}/3")

        # Training phase
        train_total = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image']  # Shape: [batch_size, H, W, D]
            presence_labels = batch['presence_labels']  # Shape: [batch_size, n_structures]
            scenarios = batch['scenario']  # List of scenario names

            # Your model training code would go here
            # model_output = model(images)
            # loss = criterion(model_output, presence_labels)
            # loss.backward()
            # optimizer.step()

            train_total += len(images)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"  Train batch {batch_idx}: {len(images)} samples, "
                      f"Total processed: {train_total}/{len(train_dataset)}")

            if batch_idx >= 5:  # Just process a few batches for demo
                break

        # Validation phase
        val_total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image']
                presence_labels = batch['presence_labels']

                # Your model evaluation code would go here
                # model_output = model(images)
                # val_loss = criterion(model_output, presence_labels)

                val_total += len(images)

                if batch_idx >= 2:  # Just process a few batches for demo
                    break

        print(f"  Validation: {val_total} samples processed")


# Example usage and test
import argparse
import sys
from pathlib import Path


def parse_image_size(size_str):
    """Parse image size string like '128,128,64' into tuple"""
    try:
        parts = size_str.split(',')
        if len(parts) != 3:
            raise ValueError("Image size must have exactly 3 dimensions")
        return tuple(int(x) for x in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid image size '{size_str}': {e}")


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic medical dataset for testing AI models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument(
        '--image-size',
        type=parse_image_size,
        default=(128, 128, 64),
        help='Image dimensions as Z,Y,X (e.g., "128,128,64")'
    )

    parser.add_argument(
        '--total-samples',
        type=int,
        default=1000,
        help='Total number of samples to generate'
    )

    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data for validation (0.0-1.0)'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Fraction of data for testing (0.0-1.0)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/synthetic_medical_dataset_hdf5',
        help='Output directory for the generated dataset'
    )

    # Noise parameters
    parser.add_argument(
        '--position-noise',
        type=float,
        default=0.05,
        help='Standard deviation for position noise (0.0-0.2)'
    )

    parser.add_argument(
        '--size-noise',
        type=float,
        default=0.2,
        help='Standard deviation for size variation (0.0-0.5)'
    )

    parser.add_argument(
        '--intensity-noise',
        type=float,
        default=0.15,
        help='Standard deviation for intensity variation (0.0-0.3)'
    )

    # Control options
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip dataset generation (useful for testing loading only)'
    )

    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip sample visualization'
    )

    parser.add_argument(
        '--skip-training-test',
        action='store_true',
        help='Skip training loop test'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.validation_split + args.test_split >= 1.0:
        print("Error: validation_split + test_split must be < 1.0")
        sys.exit(1)

    if args.total_samples <= 0:
        print("Error: total_samples must be positive")
        sys.exit(1)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    print(f"Synthetic Medical Dataset Generator")
    print(f"Image size: {args.image_size}")
    print(f"Total samples: {args.total_samples}")
    print(f"Validation split: {args.validation_split}")
    print(f"Test split: {args.test_split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    if not args.skip_generation:
        # Create dataset generator
        print("Initializing dataset generator...")
        generator = SyntheticDatasetGenerator(image_size=args.image_size)

        if args.verbose:
            print(f"Available anatomical structures: {list(generator.generator.anatomy_templates.keys())}")
            print(f"Number of test scenarios: {len(generator.scenarios)}")

        # Generate dataset
        print("Generating synthetic dataset...")
        dataset = generator.generate_dataset(
            total_samples=args.total_samples,
            validation_split=args.validation_split,
            test_split=args.test_split,
            position_noise=args.position_noise,
            size_noise=args.size_noise,
            intensity_noise=args.intensity_noise
        )

        # Print dataset statistics
        print("\nDataset Statistics:")
        for split in ['train', 'val', 'test']:
            if split in dataset:
                n_samples = len(dataset[split]['images'])
                print(f"  {split}: {n_samples} samples")

                if args.verbose:
                    # Count scenarios
                    scenarios = dataset[split]['scenarios']
                    from collections import Counter
                    scenario_counts = Counter(scenarios)
                    print(f"    Top scenarios: {dict(list(scenario_counts.most_common(5)))}")

                    # Presence statistics
                    presence = dataset[split]['presence_labels']
                    presence_rates = presence.mean(axis=0)
                    present_structures = [name for name, rate in zip(dataset['structure_names'], presence_rates) if
                                          rate > 0.5]
                    print(f"    Most common structures: {present_structures[:5]}")

        # Save dataset in HDF5 format
        print(f"\nSaving dataset to {args.output_dir}...")
        generator.save_dataset_hdf5(dataset, args.output_dir)
        print("Dataset saved successfully!")

        # Memory usage info
        if args.verbose:
            total_size_gb = (dataset['train']['images'].nbytes +
                             dataset['val']['images'].nbytes +
                             (dataset['test']['images'].nbytes if 'test' in dataset else 0)) / (1024 ** 3)
            print(f"Total dataset size: {total_size_gb:.2f} GB")

    else:
        print("Skipping dataset generation...")

    # Test the lazy loading
    if Path(args.output_dir).exists():
        print("\nTesting lazy loading...")

        try:
            # Create lazy dataset
            lazy_dataset = LazyMedicalDataset(
                args.output_dir,
                split='train',
                load_masks=True,
                preload_metadata=True
            )

            print(f"Lazy dataset loaded with {len(lazy_dataset)} samples")

            if args.verbose:
                # Print class distribution
                print("\nClass Distribution (Training):")
                distribution = lazy_dataset.get_class_distribution()
                for struct_name, stats in distribution.items():
                    print(f"  {struct_name}: {stats['presence_rate']:.1%} present "
                          f"({stats['present']}/{len(lazy_dataset)})")

            # Test loading a few samples
            print("\nTesting sample loading...")
            test_indices = [0, len(lazy_dataset) // 4, len(lazy_dataset) // 2]
            for i in test_indices:
                if i < len(lazy_dataset):
                    sample = lazy_dataset[i]
                    print(f"  Sample {i}: scenario='{sample['scenario']}', "
                          f"image_shape={sample['image'].shape}, "
                          f"structures_present={sample['presence_labels'].sum().item()}")

            # Visualize samples (if not skipped)
            if not args.skip_visualization:
                print("\nVisualizing samples...")

                # Find a normal case
                normal_found = False
                for i in range(min(20, len(lazy_dataset))):
                    sample = lazy_dataset[i]
                    if 'normal' in sample['scenario']:
                        print(f"Visualizing normal case: {sample['scenario']}")
                        visualize_sample(lazy_dataset, i)
                        normal_found = True
                        break

                if not normal_found:
                    print("No normal cases found in first 20 samples")

                # Find an abnormal case
                abnormal_found = False
                for i in range(min(50, len(lazy_dataset))):
                    sample = lazy_dataset[i]
                    if any(keyword in sample['scenario'] for keyword in ['nephrectomy', 'resection', 'missing']):
                        print(f"Visualizing abnormal case: {sample['scenario']}")
                        visualize_sample(lazy_dataset, i)
                        abnormal_found = True
                        break

                if not abnormal_found:
                    print("No abnormal cases found in first 50 samples")

            else:
                print("Skipping visualization...")

            # Test the training loop (if not skipped)
            if not args.skip_training_test:
                print("\nTesting training loop...")
                example_training_loop()
            else:
                print("Skipping training loop test...")

        except FileNotFoundError as e:
            print(f"Could not test lazy loading: {e}")
            print("Dataset directory not found. Run with dataset generation first!")
        except Exception as e:
            print(f"Error during testing: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    else:
        print(f"Output directory {args.output_dir} does not exist. Cannot test lazy loading.")

    print(f"\nScript completed successfully!")
    print(f"Dataset available at: {args.output_dir}")


if __name__ == "__main__":
    main()