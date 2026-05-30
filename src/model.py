# src/model.py
"""
MobileFaceNet implementation for metric face recognition.
Outputs L2-normalized embeddings ready for cosine similarity comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from loguru import logger
from typing import Optional


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + PReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthWiseBlock(nn.Module):
    """Depthwise separable convolution with optional residual connection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return x + self.block(x)
        return self.block(x)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Efficient CNN for face verification.
    ~1M parameters, optimized for real-time and metric learning.

    Input:  [B, 3, 112, 112]
    Output: [B, embedding_dim] (L2-normalized)
    """

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Backbone layers
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.conv2_dw = ConvBlock(64, 64, 3, 1, 1)
        self.conv_23 = DepthWiseBlock(64, 64, stride=2, residual=False)
        self.conv_3 = DepthWiseBlock(64, 64, stride=1, residual=True)
        self.conv_34 = DepthWiseBlock(64, 128, stride=2, residual=False)
        self.conv_4 = DepthWiseBlock(128, 128, stride=1, residual=True)
        self.conv_5 = DepthWiseBlock(128, 128, stride=1, residual=True)
        self.conv_6_sep = ConvBlock(128, 512, 1, 1, 0)

        # Global pooling + embedding head
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Robust alternative to GDC 7x7
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)

        # Embedding projection
        x = self.global_pool(x).flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        # 🔑 L2 Normalization: crucial for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        return x

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, dropout={self.dropout.p}"


def load_pretrained_backbone(
    model: MobileFaceNet, weights_path: str, device: torch.device = torch.device("cpu")
) -> MobileFaceNet:
    raw_path = Path(weights_path).expanduser()
    if not raw_path.is_absolute():
        path = (PROJECT_ROOT / raw_path).resolve()
    else:
        path = raw_path

    if not path.exists():
        logger.warning(f"⚠️ Weights not found at `{path}`. Initializing from scratch.")
        return model

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb < 1.0:
        logger.error(f"❌ Weights file too small ({file_size_mb:.2f} MB). Corrupted.")
        return model

    try:
        logger.info(f"📦 Loading weights: {path.name} ({file_size_mb:.2f} MB)")
        state_dict = torch.load(path, map_location=device, weights_only=False)

        if isinstance(state_dict, dict):
            state_dict = (
                state_dict.get("model_state")
                or state_dict.get("state_dict")
                or state_dict.get("net")
                or state_dict
            )

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 🔍 FILTER: Skip keys with shape mismatch (e.g., final BN layer 512 vs 128)
        model_state = model.state_dict()
        valid_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped_keys.append(k)
            else:
                valid_dict[k] = v

        if skipped_keys:
            logger.warning(
                f"⚠️ Skipped {len(skipped_keys)} mismatched keys (safe for metric learning): {skipped_keys[:3]}..."
            )

        model.load_state_dict(valid_dict, strict=False)
        logger.info(f"✅ Loaded pretrained weights successfully")

    except Exception as e:
        logger.error(f"❌ Failed to load pretrained weights: {e}")
        logger.warning("🔄 Continuing with random initialization.")

    return model


# ==========================
# 🔧 Quick Test Block
# ==========================
if __name__ == "__main__":
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Running on Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("💻 Running on CPU")

    # Init model
    model = MobileFaceNet(embedding_dim=128).to(device)
    logger.info(f"📐 Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 112, 112, device=device)
    with torch.no_grad():
        embeddings = model(dummy_input)

    # Verify output
    logger.info(f"📤 Output shape: {embeddings.shape}")
    logger.info(f"📏 L2 norms: {embeddings.norm(dim=1)}")  # Should be ~[1.0, 1.0]
    assert embeddings.shape == (2, 128), "Shape mismatch!"
    assert torch.allclose(
        embeddings.norm(dim=1), torch.ones(2, device=device), atol=1e-5
    ), "Not L2-normalized!"

    logger.info("✅ Model test passed. Ready for training/inference.")
