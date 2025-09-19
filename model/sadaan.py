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


class Encoder(nn.Module):
    """Improved U-Net style encoder with skip connections"""

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Encoder path
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.enc5 = self._conv_block(256, 512)

        # Decoder path
        self.up4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)  # 256 + 256 from skip

        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 128 + 128 from skip

        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)  # 64 + 64 from skip

        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)  # 32 + 32 from skip

        # Final feature layer
        self.final_conv = nn.Conv3d(32, 256, 1)

        self.pool = nn.MaxPool3d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)  # Full resolution
        e2 = self.enc2(self.pool(e1))  # 1/2
        e3 = self.enc3(self.pool(e2))  # 1/4
        e4 = self.enc4(self.pool(e3))  # 1/8
        e5 = self.enc5(self.pool(e4))  # 1/16

        # Decoder with skip connections
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        features = self.final_conv(d1)
        return features


class SegmentationHead(nn.Module):
    """Improved segmentation head with better conditional logic"""

    def __init__(self, in_channels: int, num_structures: int):
        super().__init__()
        self.num_structures = num_structures

        self.segmentation_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, num_structures, 1)
        )

    def forward(self, features: torch.Tensor, attention_outputs: Dict[str, torch.Tensor],
                presence_probs: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        # Generate raw segmentation logits
        raw_logits = self.segmentation_conv(features)  # [B, num_structures, D, H, W]

        # Apply attention gating
        attention_maps = attention_outputs['attention_maps']
        attention_gated = raw_logits * attention_maps

        # IMPROVED: Apply presence-based suppression more carefully
        # Create smooth gating instead of hard thresholding
        presence_gates = torch.sigmoid(10 * (presence_probs - threshold))  # Smooth step function
        presence_gates = presence_gates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Apply presence gating
        conditional_logits = attention_gated * presence_gates

        # For absent structures, actively suppress with negative bias
        absent_mask = (presence_probs < threshold).float()
        absent_bias = -10.0 * absent_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conditional_logits = conditional_logits + absent_bias

        return {
            'segmentation_logits': conditional_logits,
            'segmentation_probs': torch.sigmoid(conditional_logits),
            'raw_logits': raw_logits,
            'attention_gated': attention_gated
        }


class SpatialAttentionMedicalSegmenter(nn.Module):
    """Updated model with improved components"""

    def __init__(self, in_channels: int = 1, num_structures: int = 5,
                 feature_channels: int = 256, spatial_dims: Tuple[int, int, int] = (64, 64, 64)):
        super().__init__()
        self.num_structures = num_structures

        # Improved encoder
        self.encoder = Encoder(in_channels)

        self.attention_module = AnatomicalAttentionModule(
            feature_channels, num_structures, spatial_dims
        )

        self.absence_detector = AbsenceDetectionHead(num_structures, feature_channels)

        # Improved segmentation head
        self.segmentation_head = SegmentationHead(feature_channels, num_structures)

    def forward(self, x: torch.Tensor, presence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        # Extract features with improved encoder
        features = self.encoder(x)

        # Generate attention maps
        attention_outputs = self.attention_module(features)

        # Predict structure presence/absence
        absence_outputs = self.absence_detector(features, attention_outputs)

        # Improved conditional segmentation
        segmentation_outputs = self.segmentation_head(
            features, attention_outputs, absence_outputs['presence_probs'], presence_threshold
        )

        return {
            **attention_outputs,
            **absence_outputs,
            **segmentation_outputs
        }


class SpatialAttentionLoss(nn.Module):
    """Fixed loss function with proper handling of absent structures"""

    def __init__(self, structure_names, weights=None, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.structure_names = structure_names
        self.num_structures = len(structure_names)

        default_weights = {
            'segmentation': 1.0,
            'absence': 2.0,  # Increased weight for presence detection
            'attention_supervision': 0.3,
            'confidence': 0.1,
            'dice': 1.0,
            'false_positive_suppression': 0.5  # New loss component
        }

        self.weights = weights if weights else default_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')

    def dice_loss(self, probs, targets, epsilon=1e-6):
        """Dice loss with proper handling of empty masks"""
        # Flatten spatial dimensions
        probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W*D]
        targets_flat = targets.reshape(targets.size(0), targets.size(1), -1)

        intersection = (probs_flat * targets_flat).sum(dim=-1)  # [B, C]
        dice_scores = (2.0 * intersection + epsilon) / (
                probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1) + epsilon
        )

        return 1.0 - dice_scores.mean()

    def forward(self, outputs, targets):
        seg_logits = outputs['segmentation_logits']
        seg_probs = outputs['segmentation_probs']
        presence_probs = outputs['presence_probs']

        seg_targets = targets['segmentation_targets'].float()
        presence_targets = targets['presence_targets'].float()

        batch_size = seg_logits.shape[0]
        device = seg_logits.device
        losses = {}

        # 1. SEGMENTATION LOSS (for present structures only)
        present_mask = presence_targets.bool()  # [B, num_structures]

        seg_loss_total = 0
        dice_loss_total = 0
        valid_samples = 0

        for b in range(batch_size):
            for s in range(self.num_structures):
                if present_mask[b, s]:
                    # Standard BCE loss
                    seg_bce = self.bce_with_logits(
                        seg_logits[b, s], seg_targets[b, s]
                    ).mean()

                    # Dice loss
                    dice = self.dice_loss(
                        seg_probs[b:b + 1, s:s + 1],
                        seg_targets[b:b + 1, s:s + 1]
                    )

                    seg_loss_total += seg_bce
                    dice_loss_total += dice
                    valid_samples += 1

        if valid_samples > 0:
            losses['segmentation'] = seg_loss_total / valid_samples
            losses['dice'] = dice_loss_total / valid_samples
        else:
            losses['segmentation'] = torch.tensor(0.0, device=device)
            losses['dice'] = torch.tensor(0.0, device=device)

        # 2. FALSE POSITIVE SUPPRESSION (for absent structures)
        # Penalize any positive predictions when structure is absent
        fp_loss = 0
        fp_count = 0

        for b in range(batch_size):
            for s in range(self.num_structures):
                if not present_mask[b, s]:  # Structure is absent
                    # Penalize positive predictions
                    fp_penalty = torch.clamp(seg_probs[b, s], min=0).mean()
                    fp_loss += fp_penalty
                    fp_count += 1

        if fp_count > 0:
            losses['false_positive_suppression'] = fp_loss / fp_count
        else:
            losses['false_positive_suppression'] = torch.tensor(0.0, device=device)

        # 3. PRESENCE DETECTION LOSS
        presence_loss = self.bce(presence_probs, presence_targets).mean()
        losses['absence'] = presence_loss

        # 4. ATTENTION SUPERVISION
        if 'attention_maps' in outputs:
            attention_maps = outputs['attention_maps']
            attention_loss = 0

            for b in range(batch_size):
                for s in range(self.num_structures):
                    if present_mask[b, s]:
                        # Attention should align with ground truth mask
                        target_attention = (seg_targets[b, s] > 0.5).float()
                        attention_bce = self.bce(
                            attention_maps[b, s], target_attention
                        ).mean()
                        attention_loss += attention_bce

            losses['attention_supervision'] = attention_loss / (batch_size * self.num_structures)

        # 5. CONFIDENCE REGULARIZATION
        confidence_scores = outputs.get('confidence_scores', torch.ones_like(presence_probs))
        confidence_loss = 0

        for b in range(batch_size):
            for s in range(self.num_structures):
                pred_conf = presence_probs[b, s]
                true_pres = presence_targets[b, s]

                # Penalize overconfident wrong predictions
                if true_pres == 0 and pred_conf > 0.5:
                    confidence_loss += (pred_conf - 0.5) ** 2
                elif true_pres == 1 and pred_conf < 0.5:
                    confidence_loss += (0.5 - pred_conf) ** 2

        losses['confidence'] = confidence_loss / (batch_size * self.num_structures)

        # TOTAL LOSS
        total_loss = (
                self.weights['segmentation'] * losses['segmentation'] +
                self.weights['dice'] * losses['dice'] +
                self.weights['absence'] * losses['absence'] +
                self.weights.get('attention_supervision', 0.3) * losses.get('attention_supervision', 0) +
                self.weights['confidence'] * losses['confidence'] +
                self.weights['false_positive_suppression'] * losses['false_positive_suppression']
        )

        losses['total'] = total_loss
        return losses


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


# Usage example with debugging
def debug_model():
    """Function to test and debug the improved model"""

    # Initialize improved components
    structure_names = ['left_kidney', 'right_kidney', 'liver', 'spleen', 'pancreas']

    # Use improved model
    model = SpatialAttentionMedicalSegmenter(
        in_channels=1,
        num_structures=len(structure_names),
        spatial_dims=(64, 64, 64)
    )

    # Use improved loss
    loss_fn = SpatialAttentionLoss(structure_names)

    # Test with sample data
    batch_size, spatial_dims = 2, (64, 64, 64)
    x = torch.randn(batch_size, 1, *spatial_dims)

    # Create more realistic targets
    targets = {
        'segmentation_targets': torch.randint(0, 2, (batch_size, len(structure_names), *spatial_dims)).float(),
        'presence_targets': torch.randint(0, 2, (batch_size, len(structure_names))).float(),
    }

    model.eval()
    with torch.no_grad():
        outputs = model(x)

        print("=== IMPROVED MODEL DEBUG ===")
        print(f"Input shape: {x.shape}")

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}, mean: {value.mean():.6f}, std: {value.std():.6f}")

        # Test loss computation
        losses = loss_fn(outputs, targets)
        print(f"\n=== LOSS BREAKDOWN ===")
        for key, value in losses.items():
            print(f"{key}: {value.item():.6f}")

        # Check for common issues
        print(f"\n=== DIAGNOSTIC CHECKS ===")
        seg_probs = outputs['segmentation_probs']
        presence_probs = outputs['presence_probs']

        print(f"Segmentation prob range: [{seg_probs.min():.6f}, {seg_probs.max():.6f}]")
        print(f"Presence prob range: [{presence_probs.min():.6f}, {presence_probs.max():.6f}]")
        print(f"Non-zero segmentation pixels: {(seg_probs > 0.1).sum().item()}")

        # Check presence vs segmentation consistency
        for i, struct_name in enumerate(structure_names):
            pres_prob = presence_probs[0, i].item()
            seg_mean = seg_probs[0, i].mean().item()
            print(f"{struct_name}: presence={pres_prob:.3f}, seg_mean={seg_mean:.6f}")


if __name__ == "__main__":
    debug_model()
