# Prototyping

## Overview

* [model](https://github.com/dr-darryl-wright/sadaan/tree/main/model)
* [synthetic data generation](https://github.com/dr-darryl-wright/sadaan/tree/main/utils)
* [training protocol](https://github.com/dr-darryl-wright/sadaan/tree/main/train)
* [testing](https://github.com/dr-darryl-wright/sadaan/tree/main/test)
* [analysis](https://github.com/dr-darryl-wright/sadaan/tree/main/analysis)
---
## Replication

### data generation
```bash
$ python synthetic_medical_generator.py \
    --image-size "64,64,32" \
    --total-samples 1000 \
    --output-dir "../data/synthetic_medical_dataset_hdf5/"
```

### training
```bash
$ python train.py with dataset_path="../data/synthetic_medical_dataset_hdf5/" training.batch_size="4"
```
**loss**
![](/src/loss.png)

**metrics**
![](/src/metrics.png)

### testing
```bash
$ python test.py --weights ../train/checkpoints/best_model.pth --dataset ../data/synthetic_medical_dataset_hdf5/ --split test --output ./revised_backbone/test/
```

### analysis
```bash
$ python analysis.py --train ../test/revised_backbone/train/detailed_metrics.json --val ../test/revised_backbone/val/detailed_metrics.json --test ../test/revised_backbone/test/detailed_metrics.json -o ./revised_backbone/
```
---

## Analysis

**1. OVERALL PERFORMANCE:**

TRAIN: Presence Acc: 0.9994, Dice: 0.9338, IoU: 0.8862

VALIDATION: Presence Acc: 0.9731, Dice: 0.8394, IoU: 0.7741

TEST: Presence Acc: 0.9650, Dice: 0.8396, IoU: 0.7762

**2. STRUCTURE PERFORMANCE (Test Set):**

Best performing structures (Dice score):
            liver: 0.9803
       right_lung: 0.9791
        left_lung: 0.9718

Worst performing structures (Dice score):
               L4: 0.7216
               L3: 0.6904
               L2: 0.6042

### Present/Absent
![](/analysis/revised_backbone/presence_accuracy_comparison.png)

![](/analysis/revised_backbone/performance_heatmap.png)

### Scenario analysis
![](/analysis/revised_backbone/scenario_based_analysis.png)

### Segmentation
![](/analysis/revised_backbone/segmentation_dice_comparison.png)

### Test set example visualisation
![](/test/revised_backbone/test/comprehensive_analysis_sample_1.png)
---
---
![](/test/revised_backbone/test/comprehensive_analysis_sample_2.png)
---
---
![](/test/revised_backbone/test/comprehensive_analysis_sample_3.png)
---
---
![](/test/revised_backbone/test/comprehensive_analysis_sample_4.png)
---
---
![](/test/revised_backbone/test/comprehensive_analysis_sample_5.png)