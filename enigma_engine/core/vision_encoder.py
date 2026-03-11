"""
Vision Encoder — A small Vision Transformer (ViT) built from scratch.

No pretrained weights, no external downloads. Trains from scratch on user data
alongside the text model's projection layer. Uses the same components as the
text transformer (RMSNorm, Attention, FeedForward, TransformerBlock pattern).

Architecture:
    Image → PatchEmbedding → + Position Embeddings → N TransformerBlocks → RMSNorm → features

Input:  [batch, 3, image_size, image_size] image tensor
Output: [batch, num_patches, dim] feature tensor

Size presets:
    tiny   — 2 layers, 128-dim (~500K params)
    small  — 4 layers, 256-dim (~4M params)  [default]
    medium — 6 layers, 512-dim (~25M params)

Usage:
    from enigma_engine.core.vision_encoder import (
        VisionEncoder, VisionEncoderConfig, VISION_PRESETS,
        preprocess_image, encode_image,
    )

    encoder = VisionEncoder(VISION_PRESETS["small"])
    features = encode_image(encoder, "photo.png")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image as _PILImage

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisionEncoderConfig:
    """
    Configuration for the Vision Encoder.

    Fields:
        image_size:  Input image resolution (square). Images are resized to this.
        patch_size:  Size of each image patch. image_size must be divisible by this.
        channels:    Number of input channels (3 for RGB).
        dim:         Hidden dimension of the transformer.
        n_layers:    Number of transformer blocks.
        n_heads:     Number of attention heads.
        dropout:     Dropout rate during training.
        use_rms_norm: Use RMSNorm (True) or LayerNorm (False).
        use_cnn_stem: Use hybrid CNN+ViT (V-G). When True, a CNN stem
            extracts spatial features (edges, textures, shapes) before
            the transformer processes global relationships. This gives
            ViTs spatial inductive bias that pure attention must learn
            from data, making training more efficient on small datasets.
    """
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    use_rms_norm: bool = True
    use_cnn_stem: bool = False

    @property
    def num_patches(self) -> int:
        """Number of patches the image gets split into."""
        return (self.image_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        """Convert config to a plain dict for serialization."""
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "channels": self.channels,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "use_rms_norm": self.use_rms_norm,
            "use_cnn_stem": self.use_cnn_stem,
        }


# =============================================================================
# SIZE PRESETS
# =============================================================================

VISION_PRESETS: dict[str, VisionEncoderConfig] = {
    "tiny": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=128, n_layers=2, n_heads=4,
    ),
    "small": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=256, n_layers=4, n_heads=4,
    ),
    "medium": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=512, n_layers=6, n_heads=8,
    ),
    # V-G: Hybrid CNN+ViT presets — CNN stem provides spatial features
    "hybrid_small": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=256, n_layers=4, n_heads=4,
        use_cnn_stem=True,
    ),
    "hybrid_medium": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=512, n_layers=6, n_heads=8,
        use_cnn_stem=True,
    ),
}


# =============================================================================
# COMPONENTS
# =============================================================================

class _RMSNorm(nn.Module):
    """RMSNorm — same as model_components.RMSNorm but standalone to avoid circular imports."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerically stable norm computation
        orig_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms).to(orig_dtype) * self.weight


class CNNStem(nn.Module):
    """CNN stem for hybrid CNN+ViT architecture (V-G).

    Provides spatial inductive bias that pure ViTs lack — CNN layers
    learn edge/texture/shape hierarchies from their convolutional
    structure, while transformers excel at global reasoning.

    Architecture (3 stages):
        [B, 3, H, W] → Conv1(3→C1, k=3, s=2) → BN → GELU → pool
                      → Conv2(C1→C2, k=3, s=1) → BN → GELU → pool
                      → Conv3(C2→dim, k=3, s=1) → BN → GELU
                      → flatten → [B, num_patches, dim]

    The CNN replaces direct patch embedding, giving the ViT access
    to pre-processed spatial features instead of raw pixel patches.
    This provides the spatial hierarchy (edges → textures → shapes)
    that pure attention must learn from data.
    """

    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        # Progressive channel expansion: 3 → dim//4 → dim//2 → dim
        c1 = max(32, dim // 4)
        c2 = max(64, dim // 2)

        self.stage1 = nn.Sequential(
            nn.Conv2d(channels, c1, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image through CNN stages.

        Args:
            x: [B, C, H, W] image tensor.

        Returns:
            [B, num_patches, dim] feature tensor (spatial dims flattened).
        """
        x = self.stage1(x)   # [B, c1, H/4, W/4]
        x = self.stage2(x)   # [B, c2, H/16, W/16]
        x = self.stage3(x)   # [B, dim, H/16, W/16]
        # Flatten spatial dims: [B, dim, H', W'] → [B, H'*W', dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.

    Uses a single Conv2d with kernel_size=patch_size and stride=patch_size
    to chop the image into non-overlapping patches and project each patch
    to the transformer hidden dimension.

    Input:  [batch, channels, H, W]
    Output: [batch, num_patches, dim]
    """

    def __init__(self, patch_size: int, channels: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            channels, dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → conv: [B, dim, H/P, W/P] → flatten: [B, num_patches, dim]
        x = self.proj(x)                        # [B, dim, H/P, W/P]
        x = x.flatten(2)                         # [B, dim, num_patches]
        x = x.transpose(1, 2)                    # [B, num_patches, dim]
        return x


class _VisionAttention(nn.Module):
    """
    Multi-head self-attention for vision patches.

    Simpler than the text Attention — no KV cache, no RoPE, no GQA.
    Vision patches don't need positional rotation (we use learned position
    embeddings instead) and there's no autoregressive generation.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project to Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)                  # each: [B, heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class _VisionFeedForward(nn.Module):
    """
    Feed-forward network for vision transformer blocks.

    Uses SwiGLU activation (same as text model) for better learning:
    FFN(x) = (SiLU(W1·x) * W3·x) · W2
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * 4 * 2 / 3)  # SwiGLU convention: 8/3 * dim
        # Round to nearest multiple of 8 for hardware efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class _VisionBlock(nn.Module):
    """
    Single vision transformer block: Norm → Attention → + → Norm → FFN → +

    Same pre-norm architecture as the text transformer blocks.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 use_rms_norm: bool = True) -> None:
        super().__init__()
        Norm = _RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = Norm(dim)
        self.attn = _VisionAttention(dim, n_heads, dropout)
        self.norm2 = Norm(dim)
        self.ffn = _VisionFeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MAIN ENCODER
# =============================================================================

class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder built from scratch.

    Architecture (pure ViT):
        Image → PatchEmbedding → + pos_embed → VisionBlocks × N → RMSNorm → features

    Architecture (hybrid CNN+ViT, when use_cnn_stem=True):
        Image → CNNStem → + pos_embed → VisionBlocks × N → RMSNorm → features

    The hybrid architecture (V-G) adds CNN layers before the transformer.
    CNN learns spatial hierarchy (edges → textures → shapes) that pure ViT
    needs far more training data to learn.  The transformer then handles
    global reasoning across the CNN-extracted features.

    Trains from scratch alongside the text model's vision_projection layer.
    Quality depends entirely on user's training data and compute.

    Args:
        config: VisionEncoderConfig with architecture settings.
    """

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # V-G: Hybrid CNN+ViT uses CNN stem instead of patch embedding
        if config.use_cnn_stem:
            self.cnn_stem = CNNStem(
                channels=config.channels,
                dim=config.dim,
            )
            self.patch_embed = None
            # CNN stem output spatial size: image_size / 8
            # (2× from stride-2 conv × 2× from pool1 × 2× from pool2)
            cnn_patches = (config.image_size // 8) ** 2
            self.pos_embed = nn.Parameter(
                torch.randn(1, cnn_patches, config.dim) * 0.02
            )
            logger.info(
                "VisionEncoder: hybrid CNN+ViT mode "
                "(CNN stem → %d patches → %d transformer blocks)",
                cnn_patches, config.n_layers)
        else:
            self.cnn_stem = None
            # Patch embedding: image → patch tokens
            self.patch_embed = PatchEmbedding(
                patch_size=config.patch_size,
                channels=config.channels,
                dim=config.dim,
            )
            # Learnable position embeddings for each patch
            self.pos_embed = nn.Parameter(
                torch.randn(1, config.num_patches, config.dim) * 0.02
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _VisionBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_rms_norm=config.use_rms_norm,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization
        Norm = _RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.norm = Norm(config.dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor into patch features.

        Args:
            x: Image tensor [batch, channels, image_size, image_size]

        Returns:
            Feature tensor [batch, num_patches, dim]
        """
        # V-G: Hybrid CNN+ViT uses CNN stem for spatial features
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)         # [B, cnn_patches, dim]
        else:
            x = self.patch_embed(x)      # [B, num_patches, dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Transform through blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def _ensure_pil() -> None:
    """Ensure PIL is available, raise ImportError with helpful message if not."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install with: pip install Pillow"
        ) from None


def preprocess_image(
    image: Union[str, Path, "_PILImage"],
    image_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess an image for the vision encoder.

    Handles:
    - File paths (str or Path) → loads via PIL
    - PIL Image objects → converts directly
    - Grayscale/RGBA → converts to RGB
    - Any size → resizes to image_size × image_size
    - Normalizes pixel values to [-1, 1]

    Args:
        image: PIL Image, file path string, or Path object.
        image_size: Target square size (default 224).

    Returns:
        Tensor of shape [1, 3, image_size, image_size] normalized to [-1, 1].
    """
    _ensure_pil()
    from PIL import Image

    # Load from path if needed
    if isinstance(image, (str, Path)):
        image = Image.open(str(image))

    # Convert to RGB (handles grayscale, RGBA, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to target size
    image = image.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor: HWC uint8 → CHW float [-1, 1]
    import numpy as np
    arr = np.array(image, dtype=np.float32)      # [H, W, 3] in [0, 255]
    arr = arr / 127.5 - 1.0                        # normalize to [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    return tensor.unsqueeze(0)  # [1, 3, H, W]


# =============================================================================
# CONVENIENCE ENCODE FUNCTIONS
# =============================================================================

@torch.no_grad()
def encode_image(
    encoder: VisionEncoder,
    image: Union[str, Path, "_PILImage"],
) -> torch.Tensor:
    """
    Preprocess and encode a single image.

    Args:
        encoder: Trained VisionEncoder instance.
        image: PIL Image, file path, or Path.

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    tensor = preprocess_image(image, image_size=encoder.config.image_size)
    tensor = tensor.to(next(encoder.parameters()).device)
    encoder.eval()
    return encoder(tensor)


@torch.no_grad()
def encode_video_frames(
    encoder: VisionEncoder,
    video_path: Union[str, Path],
    max_frames: int = 8,
    max_visual_tokens: int = 0,
    dedup_threshold: float = 0.95,
) -> torch.Tensor:
    """
    Sample frames from a video and encode each one.

    Requires OpenCV (cv2). Samples evenly-spaced frames, drops
    near-duplicate frames (cosine similarity > dedup_threshold),
    and optionally truncates to max_visual_tokens.

    Args:
        encoder: Trained VisionEncoder instance.
        video_path: Path to video file.
        max_frames: Maximum number of frames to sample.
        max_visual_tokens: Maximum total tokens (patches) to return.
            0 means no limit.
        dedup_threshold: Cosine similarity threshold for dropping
            duplicate frames. Set to 1.0 to disable dedup.

    Returns:
        Feature tensor [1, N*num_patches, dim] (concatenated frames).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video encoding. "
            "Install with: pip install opencv-python"
        ) from None

    _ensure_pil()
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Sample evenly-spaced frame indices
    n_sample = min(max_frames, total_frames)
    indices = [int(i * total_frames / n_sample) for i in range(n_sample)]

    device = next(encoder.parameters()).device
    encoder.eval()
    all_features = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = preprocess_image(pil_img, image_size=encoder.config.image_size)
        tensor = tensor.to(device)
        features = encoder(tensor)  # [1, num_patches, dim]
        all_features.append(features)

    cap.release()

    if not all_features:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Drop near-duplicate consecutive frames (cosine similarity)
    if dedup_threshold < 1.0 and len(all_features) > 1:
        unique: list[torch.Tensor] = [all_features[0]]
        for feat in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = feat.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < dedup_threshold:
                unique.append(feat)
        all_features = unique

    # Concatenate all frame features along the sequence dimension
    combined = torch.cat(all_features, dim=1)  # [1, N*num_patches, dim]

    # Truncate to max_visual_tokens if specified
    if max_visual_tokens > 0 and combined.shape[1] > max_visual_tokens:
        combined = combined[:, :max_visual_tokens, :]

    return combined


@torch.no_grad()
def encode_screen(encoder: VisionEncoder) -> torch.Tensor:
    """
    Capture the screen and encode it.

    Uses PIL.ImageGrab for screen capture (works on Windows/macOS).

    Args:
        encoder: Trained VisionEncoder instance.

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    _ensure_pil()
    from PIL import ImageGrab

    screenshot = ImageGrab.grab()
    return encode_image(encoder, screenshot)


@torch.no_grad()
def encode_camera(
    encoder: VisionEncoder,
    device_id: int = 0,
) -> torch.Tensor:
    """
    Capture a frame from the webcam and encode it.

    Requires OpenCV (cv2).

    Args:
        encoder: Trained VisionEncoder instance.
        device_id: Camera device index (default 0).

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for camera capture. "
            "Install with: pip install opencv-python"
        ) from None

    _ensure_pil()
    from PIL import Image

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device_id}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to capture frame from camera {device_id}")

    # BGR → RGB → PIL → encode
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return encode_image(encoder, pil_img)
