import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class AnatomicalAttentionModule(nn.Module):
    """
    Learns spatial attention maps for specific anatomical structures
    based on expected anatomical locations and relationships.
    """

    def __init__(self, in_channels: int, num_structures: int, spatial_dims: Tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.num_structures = num_structures
        self.spatial_dims = spatial_dims

        # Learnable anatomical position priors
        self.position_embeddings = nn.Parameter(
            torch.randn(num_structures, *spatial_dims) * 0.1
        )

        # Attention generation network
        self.attention_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, num_structures, 1),
            nn.Sigmoid()
        )

        # Feature-attention matching network
        self.feature_matcher = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_structures, 1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = features.shape[0]

        # Generate structure-specific attention maps
        attention_maps = self.attention_conv(features)  # [B, num_structures, D, H, W]

        # Add learned anatomical position priors
        position_priors = self.position_embeddings.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        enhanced_attention = attention_maps * torch.sigmoid(position_priors)

        # Compute feature responses in attended regions
        feature_responses = self.feature_matcher(features)  # [B, num_structures, D, H, W]

        # Compute attention-feature alignment for absence detection
        alignment_scores = torch.mean(enhanced_attention * torch.sigmoid(feature_responses), dim=(2, 3, 4))

        return {
            'attention_maps': enhanced_attention,
            'feature_responses': feature_responses,
            'alignment_scores': alignment_scores,
            'position_priors': position_priors
        }


class AbsenceDetectionHead(nn.Module):
    """
    Determines structure presence/absence based on attention-feature alignment.
    """

    def __init__(self, num_structures: int, feature_dim: int = 256):
        super().__init__()
        self.num_structures = num_structures

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Per-structure absence detection
        self.absence_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + 1, 64),  # +1 for alignment score
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2)  # present/absent
            ) for _ in range(num_structures)
        ])

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_structures),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, attention_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = features.shape[0]

        # Global feature representation
        global_features = self.global_pool(features).flatten(1)  # [B, feature_dim]

        # Per-structure absence prediction
        alignment_scores = attention_outputs['alignment_scores']  # [B, num_structures]

        absence_logits = []
        for i in range(self.num_structures):
            # Combine global features with structure-specific alignment score
            struct_input = torch.cat([global_features, alignment_scores[:, i:i + 1]], dim=1)
            struct_logits = self.absence_classifiers[i](struct_input)
            absence_logits.append(struct_logits)

        absence_logits = torch.stack(absence_logits, dim=1)  # [B, num_structures, 2]

        # Confidence estimation
        confidence_scores = self.confidence_head(global_features)  # [B, num_structures]

        return {
            'absence_logits': absence_logits,
            'presence_probs': F.softmax(absence_logits, dim=-1)[:, :, 1],  # prob of present
            'confidence_scores': confidence_scores
        }


class SegmentationHead(nn.Module):
    """
    Conditional segmentation head that only segments detected structures.
    """

    def __init__(self, in_channels: int, num_structures: int):
        super().__init__()
        self.num_structures = num_structures

        self.segmentation_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, num_structures, 1)
        )

    def forward(self, features: torch.Tensor, attention_outputs: Dict[str, torch.Tensor],
                presence_probs: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        # Generate segmentation logits
        seg_logits = self.segmentation_conv(features)  # [B, num_structures, D, H, W]

        # Apply attention-based gating
        attention_maps = attention_outputs['attention_maps']
        gated_logits = seg_logits * attention_maps

        # Suppress segmentation for structures detected as absent
        presence_mask = (presence_probs > threshold).float()  # [B, num_structures]
        presence_mask = presence_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conditional_logits = gated_logits * presence_mask

        return {
            'segmentation_logits': conditional_logits,
            'segmentation_probs': torch.sigmoid(conditional_logits),
            'raw_logits': seg_logits
        }


class SpatialAttentionMedicalSegmenter(nn.Module):
    """
    Complete model combining spatial attention, absence detection, and conditional segmentation.
    """

    def __init__(self, in_channels: int = 1, num_structures: int = 5,
                 feature_channels: int = 256, spatial_dims: Tuple[int, int, int] = (64, 64, 64)):
        super().__init__()
        self.num_structures = num_structures

        # Feature extraction backbone (simplified U-Net style)
        self.encoder = self._build_encoder(in_channels, feature_channels)

        # Spatial attention module
        self.attention_module = AnatomicalAttentionModule(
            feature_channels, num_structures, spatial_dims
        )

        # Absence detection head
        self.absence_detector = AbsenceDetectionHead(num_structures, feature_channels)

        # Conditional segmentation head
        self.segmentation_head = SegmentationHead(feature_channels, num_structures)

    def _build_encoder(self, in_channels: int, out_channels: int) -> nn.Module:
        """Simplified encoder - replace with your preferred backbone"""
        return nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x: torch.Tensor, presence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.encoder(x)  # [B, feature_channels, D, H, W]

        # Generate attention maps and detect anatomical structures
        attention_outputs = self.attention_module(features)

        # Predict structure presence/absence
        absence_outputs = self.absence_detector(features, attention_outputs)

        # Conditional segmentation
        segmentation_outputs = self.segmentation_head(
            features, attention_outputs, absence_outputs['presence_probs'], presence_threshold
        )

        return {
            **attention_outputs,
            **absence_outputs,
            **segmentation_outputs
        }


class SpatialAttentionLoss(nn.Module):
    """
    Combined loss function for training the spatial attention model.
    """

    def __init__(self, structure_names: List[str], weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.structure_names = structure_names
        self.num_structures = len(structure_names)

        # Loss weights
        default_weights = {
            'segmentation': 1.0,
            'absence': 1.0,
            'attention_supervision': 0.5,
            'confidence': 0.1
        }
        self.weights = weights if weights is not None else default_weights

        # Loss functions
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.absence_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}

        # Segmentation loss (only for present structures)
        if 'segmentation_targets' in targets and 'presence_targets' in targets:
            seg_logits = outputs['segmentation_logits']
            seg_targets = targets['segmentation_targets']
            presence_targets = targets['presence_targets']  # [B, num_structures]

            # Mask loss to only include present structures
            seg_loss = 0
            for i in range(self.num_structures):
                present_mask = presence_targets[:, i].bool()
                if present_mask.any():
                    struct_logits = seg_logits[present_mask, i]
                    struct_targets = seg_targets[present_mask, i]
                    seg_loss += self.seg_loss(struct_logits, struct_targets.float())

            losses['segmentation'] = seg_loss / max(1, self.num_structures)

        # Absence detection loss
        if 'presence_targets' in targets:
            absence_logits = outputs['absence_logits']  # [B, num_structures, 2]
            presence_targets = targets['presence_targets'].long()  # [B, num_structures]

            absence_loss = 0
            for i in range(self.num_structures):
                absence_loss += self.absence_loss(absence_logits[:, i], presence_targets[:, i])
            losses['absence'] = absence_loss / self.num_structures

        # Attention supervision loss (if attention targets provided)
        if 'attention_targets' in targets:
            attention_maps = outputs['attention_maps']
            attention_targets = targets['attention_targets']
            losses['attention_supervision'] = self.mse_loss(attention_maps, attention_targets)

        # Confidence calibration loss
        if 'presence_targets' in targets:
            confidence_scores = outputs['confidence_scores']
            presence_probs = outputs['presence_probs']
            presence_targets_float = targets['presence_targets'].float()

            # Encourage high confidence when predictions are correct
            correct_predictions = (presence_probs > 0.5) == presence_targets_float
            confidence_targets = correct_predictions.float()
            losses['confidence'] = self.mse_loss(confidence_scores, confidence_targets)

        # Combine losses
        total_loss = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        losses['total'] = total_loss

        return losses


# Example usage and training setup
def create_sample_data(batch_size: int = 2, num_structures: int = 5):
    """Create sample data for demonstration"""
    spatial_dims = (64, 64, 64)

    # Sample input image
    x = torch.randn(batch_size, 1, *spatial_dims)

    # Sample targets
    targets = {
        'segmentation_targets': torch.randint(0, 2, (batch_size, num_structures, *spatial_dims)),
        'presence_targets': torch.randint(0, 2, (batch_size, num_structures)),
        'attention_targets': torch.rand(batch_size, num_structures, *spatial_dims)
    }

    return x, targets


if __name__ == "__main__":
    # Initialize model
    structure_names = ['left_kidney', 'right_kidney', 'liver', 'spleen', 'pancreas']
    model = SpatialAttentionMedicalSegmenter(
        in_channels=1,
        num_structures=len(structure_names),
        spatial_dims=(64, 64, 64)
    )

    # Initialize loss function
    loss_fn = SpatialAttentionLoss(structure_names)

    # Create sample data
    x, targets = create_sample_data(batch_size=2, num_structures=len(structure_names))

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)

        print("Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

        # Compute loss
        losses = loss_fn(outputs, targets)
        print(f"\nLoss components:")
        for key, value in losses.items():
            print(f"  {key}: {value.item():.4f}")

        # Example inference
        print(f"\nExample predictions:")
        presence_probs = outputs['presence_probs'][0]  # First sample
        for i, structure in enumerate(structure_names):
            prob = presence_probs[i].item()
            confidence = outputs['confidence_scores'][0, i].item()
            status = "PRESENT" if prob > 0.5 else "ABSENT"
            print(f"  {structure}: {status} (prob={prob:.3f}, confidence={confidence:.3f})")