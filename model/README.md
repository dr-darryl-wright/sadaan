# SADAAn (Spatial Attention for Detection of Absent Anatomy)

## Spatial Attention for Absence Detection
**Mechanism**: Train attention maps to highlight where structures should be located based on anatomical knowledge, then use the mismatch between expected and observed features to detect absence.
**Implementation**:

* **Anatomical Attention Maps**: Learn position-specific attention that encodes where each structure typically appears (e.g., kidneys in the retroperitoneum, L4 vertebra relative to other spine levels)
* **Feature-Attention Divergence**: When expected anatomical features are absent in high-attention regions, this signals missing structures
* **Attention Suppression**: Train the model to actively suppress attention in regions where structures are confirmed absent

**Intuition**: For kidney segmentation, the model learns that kidneys should receive high attention in specific bilateral locations. If one kidney is surgically absent, the attention map would show high expected attention but low feature response, triggering an "absent" classification.

## Key Components
1. **AnatomicalAttentionModule**:

Learns spatial attention maps specific to each anatomical structure
Uses learnable position embeddings that encode where structures typically appear
Computes alignment between expected attention and actual feature responses
Low alignment scores indicate potential absence

2. **AbsenceDetectionHead**:

Takes alignment scores and global features to predict presence/absence
Includes confidence estimation to indicate prediction reliability
Uses separate classifiers for each structure to handle their unique characteristics

3. **SegmentationHead**:

Applies conditional segmentation that respects presence predictions
Uses attention maps to focus segmentation on relevant regions
Suppresses segmentation for structures predicted as absent

4. **Combined Loss Function**:

Segmentation loss only applied to structures marked as present
Absence detection loss for learning presence/absence classification
Optional attention supervision if you have expert-annotated attention maps
Confidence calibration to improve reliability estimates