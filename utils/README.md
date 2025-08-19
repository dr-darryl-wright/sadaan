# Synthetic Medical Data Generation

## Key Features
**Anatomical Realism**:

* 12+ anatomical structures with realistic relative positions and sizes
* Different shape types (ellipsoids for organs, kidney shapes, cylinders for vertebrae)
* Proper anatomical relationships (bilateral structures, organ positioning)

**Controlled Variability**:

* Position, size, and intensity variations to simulate natural anatomical variation
* Imaging artifacts (noise, blur, bias fields) to mimic real medical imaging
* Multiple absence scenarios from simple (single organ) to complex (multiple organs)

**Comprehensive Testing Scenarios**:

* Normal complete anatomy (30% of data)
* Single organ missing (surgical cases like nephrectomy, splenectomy)
* Bilateral missing (both kidneys, both lungs)
* Vertebral missing (L4 missing, multiple vertebrae)
* Complex multi-organ absence patterns

**Ready-to-Use Output**:

* Train/validation/test splits with stratification
* Proper data structures for your PyTorch model
* Visualization tools for debugging
* Save/load functionality

## Usage Example

```python
# Quick start - generate dataset
generator = SyntheticDatasetGenerator(image_size=(128, 128, 64))
dataset = generator.generate_dataset(
    total_samples=1000,
    validation_split=0.2,
    test_split=0.1
)

# Access training data
train_images = dataset['train']['images']        # [N, 128, 128, 64]
train_masks = dataset['train']['masks']          # [N, 12, 128, 128, 64] 
train_presence = dataset['train']['presence_labels']  # [N, 12] - binary presence

# Structure names
structure_names = dataset['structure_names']  # ['left_lung', 'right_lung', ...]
```

## Testing Strategy
The pipeline generates scenarios that will test your model's ability to:

1. **Detect obvious absences** (missing large organs like liver)
2. **Handle bilateral cases** (both kidneys missing)
3. **Recognize small structure** absences (single vertebra missing)
4. **Deal with complex scenarios** (multiple organs missing)
5. **Maintain spatial reasoning** (attention maps should focus on correct anatomical regions)

### Advantages for Development
* **Rapid Iteration**: Generate new datasets quickly with different parameters
* **Controlled Testing**: Know exactly what should be present/absent for debugging
* **Progressive Difficulty**: Start with easy cases, gradually add complexity
* **Attention Validation**: Easy to verify if attention maps focus on anatomically correct regions

The synthetic data will let you validate your attention mechanisms, absence detection logic, and conditional segmentation before moving to real medical imaging data. You can easily adjust the complexity by changing noise levels, position variations, or the number of structures.