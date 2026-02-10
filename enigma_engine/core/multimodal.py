"""
Multimodal Support

Unified interface for text, image, and audio processing.
Enables vision-language models and multimodal embeddings.

FILE: enigma_engine/core/multimodal.py
TYPE: Core AI
MAIN CLASSES: MultimodalProcessor, ImageEncoder, AudioEncoder, MultimodalModel
"""

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalInput:
    """Container for multimodal inputs."""
    text: Optional[str] = None
    images: list[Any] = field(default_factory=list)  # PIL Images or paths
    audio: list[Any] = field(default_factory=list)  # Audio arrays or paths
    video: list[Any] = field(default_factory=list)  # Video frames
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedInput:
    """Encoded multimodal features."""
    embeddings: Tensor  # Combined embeddings [seq_len, hidden_dim]
    attention_mask: Tensor
    modality_ids: Tensor  # Which modality each token belongs to
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)


class ImageEncoder(nn.Module):
    """
    Encode images into embeddings compatible with LLM.
    
    Uses a vision transformer to extract patch embeddings,
    then projects them to the LLM hidden dimension.
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 patch_size: int = 14,
                 image_size: int = 224,
                 num_layers: int = 12,
                 num_heads: int = 12):
        """
        Initialize image encoder.
        
        Args:
            hidden_dim: Output embedding dimension
            patch_size: Size of image patches
            image_size: Input image size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            
        Returns:
            Image embeddings [batch, num_patches + 1, hidden_dim]
        """
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class AudioEncoder(nn.Module):
    """
    Encode audio into embeddings compatible with LLM.
    
    Processes mel spectrograms through a transformer.
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 n_mels: int = 80,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_frames: int = 3000):
        """
        Initialize audio encoder.
        
        Args:
            hidden_dim: Output embedding dimension
            n_mels: Number of mel bands
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_frames: Maximum audio frames
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        
        # Feature projection
        self.input_proj = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)
        
        # Subsampling (reduce sequence length)
        self.subsample = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4)
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_frames // 4, hidden_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, mel_spectrograms: Tensor) -> Tensor:
        """
        Encode audio to embeddings.
        
        Args:
            mel_spectrograms: Mel spectrogram [batch, n_mels, time]
            
        Returns:
            Audio embeddings [batch, time/4, hidden_dim]
        """
        # Feature projection
        x = self.input_proj(mel_spectrograms)  # [B, hidden_dim, T]
        
        # Subsample
        x = self.subsample(x)
        x = x.transpose(1, 2)  # [B, T/4, hidden_dim]
        
        # Add position embedding
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len]
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class ProjectionLayer(nn.Module):
    """Project modality embeddings to LLM space."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int = 2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else output_dim
            layers.extend([
                nn.Linear(in_d, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim)
            ])
        
        self.proj = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class MultimodalProcessor:
    """
    Process and combine multiple modalities.
    
    Handles preprocessing, encoding, and combining
    text, images, and audio into unified embeddings.
    """
    
    def __init__(self,
                 llm_hidden_dim: int = 768,
                 image_encoder: Optional[ImageEncoder] = None,
                 audio_encoder: Optional[AudioEncoder] = None,
                 device: str = "cpu"):
        """
        Initialize multimodal processor.
        
        Args:
            llm_hidden_dim: LLM hidden dimension for projection
            image_encoder: Custom image encoder (creates default if None)
            audio_encoder: Custom audio encoder (creates default if None)
            device: Device to run on
        """
        self.llm_hidden_dim = llm_hidden_dim
        self.device = device
        
        # Encoders
        self.image_encoder = image_encoder or ImageEncoder(hidden_dim=llm_hidden_dim)
        self.audio_encoder = audio_encoder or AudioEncoder(hidden_dim=llm_hidden_dim)
        
        # Move to device
        self.image_encoder = self.image_encoder.to(device)
        self.audio_encoder = self.audio_encoder.to(device)
        
        # Projections (if encoder dim != LLM dim)
        self.image_proj = ProjectionLayer(
            self.image_encoder.hidden_dim, llm_hidden_dim
        ).to(device)
        
        self.audio_proj = ProjectionLayer(
            self.audio_encoder.hidden_dim, llm_hidden_dim
        ).to(device)
        
        # Special tokens
        self.image_start_token = nn.Parameter(torch.randn(1, 1, llm_hidden_dim))
        self.image_end_token = nn.Parameter(torch.randn(1, 1, llm_hidden_dim))
        self.audio_start_token = nn.Parameter(torch.randn(1, 1, llm_hidden_dim))
        self.audio_end_token = nn.Parameter(torch.randn(1, 1, llm_hidden_dim))
    
    def preprocess_image(self, image) -> Tensor:
        """Preprocess image for encoding."""
        try:
            import torchvision.transforms as T
            from PIL import Image
        except ImportError:
            raise ImportError("PIL and torchvision required for image processing")
        
        transform = T.Compose([
            T.Resize((self.image_encoder.image_size, self.image_encoder.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def preprocess_audio(self, audio, sample_rate: int = 16000) -> Tensor:
        """Preprocess audio for encoding."""
        try:
            import torchaudio
            import torchaudio.transforms as T
        except ImportError:
            raise ImportError("torchaudio required for audio processing")
        
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(audio)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        else:
            waveform = torch.tensor(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        # Convert to mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=self.audio_encoder.n_mels,
            n_fft=400,
            hop_length=160
        )
        
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        
        return mel.to(self.device)
    
    def encode_images(self, images: list) -> Tensor:
        """Encode multiple images."""
        if not images:
            return None
        
        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        
        # Encode
        with torch.no_grad():
            embeddings = self.image_encoder(batch)
        
        # Project to LLM space
        embeddings = self.image_proj(embeddings)
        
        return embeddings
    
    def encode_audio(self, audio_files: list) -> Tensor:
        """Encode multiple audio files."""
        if not audio_files:
            return None
        
        embeddings_list = []
        for audio in audio_files:
            mel = self.preprocess_audio(audio)
            with torch.no_grad():
                emb = self.audio_encoder(mel)
            emb = self.audio_proj(emb)
            embeddings_list.append(emb)
        
        return torch.cat(embeddings_list, dim=1)
    
    def process(self,
                input_data: MultimodalInput,
                text_embeddings: Optional[Tensor] = None) -> EncodedInput:
        """
        Process multimodal input into combined embeddings.
        
        Args:
            input_data: Multimodal input container
            text_embeddings: Pre-computed text embeddings
            
        Returns:
            Combined encoded input
        """
        all_embeddings = []
        modality_ids = []
        positions = {}
        current_pos = 0
        
        # Process images
        if input_data.images:
            img_emb = self.encode_images(input_data.images)
            
            # Add start token
            all_embeddings.append(self.image_start_token)
            modality_ids.append(torch.full((1,), ModalityType.IMAGE.value.encode()[0]))
            current_pos += 1
            
            # Add image embeddings
            positions["images"] = (current_pos, current_pos + img_emb.shape[1])
            all_embeddings.append(img_emb)
            modality_ids.append(torch.full((img_emb.shape[1],), ModalityType.IMAGE.value.encode()[0]))
            current_pos += img_emb.shape[1]
            
            # Add end token
            all_embeddings.append(self.image_end_token)
            modality_ids.append(torch.full((1,), ModalityType.IMAGE.value.encode()[0]))
            current_pos += 1
        
        # Process audio
        if input_data.audio:
            audio_emb = self.encode_audio(input_data.audio)
            
            all_embeddings.append(self.audio_start_token)
            modality_ids.append(torch.full((1,), ModalityType.AUDIO.value.encode()[0]))
            current_pos += 1
            
            positions["audio"] = (current_pos, current_pos + audio_emb.shape[1])
            all_embeddings.append(audio_emb)
            modality_ids.append(torch.full((audio_emb.shape[1],), ModalityType.AUDIO.value.encode()[0]))
            current_pos += audio_emb.shape[1]
            
            all_embeddings.append(self.audio_end_token)
            modality_ids.append(torch.full((1,), ModalityType.AUDIO.value.encode()[0]))
            current_pos += 1
        
        # Add text embeddings
        if text_embeddings is not None:
            positions["text"] = (current_pos, current_pos + text_embeddings.shape[1])
            all_embeddings.append(text_embeddings)
            modality_ids.append(torch.full((text_embeddings.shape[1],), ModalityType.TEXT.value.encode()[0]))
        
        # Combine
        if all_embeddings:
            combined = torch.cat(all_embeddings, dim=1)
            modality_tensor = torch.cat(modality_ids)
        else:
            combined = torch.zeros(1, 0, self.llm_hidden_dim).to(self.device)
            modality_tensor = torch.zeros(0, dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones(combined.shape[:2], dtype=torch.bool).to(self.device)
        
        return EncodedInput(
            embeddings=combined,
            attention_mask=attention_mask,
            modality_ids=modality_tensor.to(self.device),
            positions=positions
        )


class MultimodalModel(nn.Module):
    """
    Multimodal language model combining vision and text.
    
    Wraps an existing LLM to accept multimodal inputs.
    """
    
    def __init__(self,
                 llm: nn.Module,
                 processor: MultimodalProcessor):
        """
        Initialize multimodal model.
        
        Args:
            llm: Base language model
            processor: Multimodal processor
        """
        super().__init__()
        
        self.llm = llm
        self.processor = processor
    
    def forward(self,
                text_input_ids: Optional[Tensor] = None,
                images: list = None,
                audio: list = None,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with multimodal inputs.
        
        Args:
            text_input_ids: Text token IDs
            images: List of images
            audio: List of audio files
            attention_mask: Attention mask
            
        Returns:
            Model outputs
        """
        # Get text embeddings from LLM
        if text_input_ids is not None:
            if hasattr(self.llm, 'get_input_embeddings'):
                text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
            else:
                text_embeddings = None
        else:
            text_embeddings = None
        
        # Process multimodal inputs
        mm_input = MultimodalInput(
            images=images or [],
            audio=audio or []
        )
        
        encoded = self.processor.process(mm_input, text_embeddings)
        
        # Forward through LLM
        return self.llm(
            inputs_embeds=encoded.embeddings,
            attention_mask=encoded.attention_mask
        )


def create_multimodal_processor(hidden_dim: int = 768,
                                device: str = "cpu") -> MultimodalProcessor:
    """Create a default multimodal processor."""
    return MultimodalProcessor(
        llm_hidden_dim=hidden_dim,
        device=device
    )


__all__ = [
    'MultimodalProcessor',
    'MultimodalInput',
    'EncodedInput',
    'ModalityType',
    'ImageEncoder',
    'AudioEncoder',
    'ProjectionLayer',
    'MultimodalModel',
    'create_multimodal_processor'
]
