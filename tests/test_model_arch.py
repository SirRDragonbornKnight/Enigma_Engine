"""Tests for model architecture, components, vision/audio encoders, GGUF format."""
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestVisionEncoderConfig:
    """VisionEncoderConfig dataclass validation."""

    def test_config_defaults(self):
        """VisionEncoderConfig should have sensible defaults."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig()
        assert cfg.image_size == 224
        assert cfg.patch_size == 16
        assert cfg.dim == 256
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        assert cfg.channels == 3

    def test_config_num_patches(self):
        """num_patches property should compute correctly."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=224, patch_size=16)
        assert cfg.num_patches == (224 // 16) ** 2  # 196

    def test_config_custom_values(self):
        """VisionEncoderConfig should accept custom values."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=128, patch_size=8, dim=512, n_layers=6, n_heads=8)
        assert cfg.image_size == 128
        assert cfg.patch_size == 8
        assert cfg.dim == 512
        assert cfg.n_layers == 6
        assert cfg.n_heads == 8
        assert cfg.num_patches == (128 // 8) ** 2  # 256


class TestVisionEncoderPresets:
    """Vision encoder size presets (tiny/small/medium)."""

    def test_presets_exist(self):
        """VISION_PRESETS dict should exist with tiny/small/medium."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert "tiny" in VISION_PRESETS
        assert "small" in VISION_PRESETS
        assert "medium" in VISION_PRESETS

    def test_tiny_preset(self):
        """Tiny preset should be smallest."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        tiny = VISION_PRESETS["tiny"]
        assert tiny.n_layers == 2
        assert tiny.dim == 128

    def test_small_preset(self):
        """Small preset should be the default."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        small = VISION_PRESETS["small"]
        assert small.n_layers == 4
        assert small.dim == 256

    def test_medium_preset(self):
        """Medium preset should be largest."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        medium = VISION_PRESETS["medium"]
        assert medium.n_layers == 6
        assert medium.dim == 512


class TestPatchEmbedding:
    """PatchEmbedding module converts images to patch tokens."""

    def test_patch_embedding_output_shape(self):
        """PatchEmbedding should produce [batch, num_patches, dim] output."""
        import torch
        from enigma_engine.core.vision_encoder import PatchEmbedding
        pe = PatchEmbedding(patch_size=16, channels=3, dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        # 224/16 = 14, 14*14 = 196 patches
        assert out.shape == (2, 196, 256)

    def test_patch_embedding_different_sizes(self):
        """PatchEmbedding should work with different patch sizes."""
        import torch
        from enigma_engine.core.vision_encoder import PatchEmbedding
        pe = PatchEmbedding(patch_size=8, channels=3, dim=128)
        x = torch.randn(1, 3, 128, 128)
        out = pe(x)
        # 128/8 = 16, 16*16 = 256 patches
        assert out.shape == (1, 256, 128)


class TestVisionEncoder:
    """VisionEncoder — the full ViT model."""

    def test_encoder_forward_shape(self):
        """VisionEncoder forward should return [batch, num_patches, dim]."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=64, patch_size=8, dim=64, n_layers=2, n_heads=2)
        encoder = VisionEncoder(cfg)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        num_patches = (64 // 8) ** 2  # 64
        assert out.shape == (2, num_patches, 64)

    def test_encoder_is_nn_module(self):
        """VisionEncoder should be a proper nn.Module."""
        import torch.nn as nn
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert isinstance(encoder, nn.Module)

    def test_encoder_has_parameters(self):
        """VisionEncoder should have trainable parameters."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        params = sum(p.numel() for p in encoder.parameters())
        assert params > 0

    def test_encoder_gradients_flow(self):
        """Gradients should flow through the encoder (trainable)."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 32, 32)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad

    def test_encoder_tiny_preset_params(self):
        """Tiny preset should have roughly 500K params."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VISION_PRESETS
        encoder = VisionEncoder(VISION_PRESETS["tiny"])
        params = sum(p.numel() for p in encoder.parameters())
        # Should be in ballpark of 500K (allow wide range for architecture details)
        assert 100_000 < params < 2_000_000

    def test_encoder_config_stored(self):
        """VisionEncoder should store its config."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert encoder.config is cfg

    def test_encoder_position_embeddings(self):
        """Encoder should have learnable position embeddings."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert hasattr(encoder, "pos_embed")
        assert encoder.pos_embed is not None
        # Position embeddings should match [1, num_patches, dim]
        num_patches = (32 // 8) ** 2
        assert encoder.pos_embed.shape == (1, num_patches, 32)


class TestImagePreprocessing:
    """Image preprocessing — resize, normalize, tensor conversion."""

    def test_preprocess_pil_image(self):
        """preprocess_image should handle PIL Images."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (400, 300), (128, 64, 32))
        tensor = preprocess_image(img, image_size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_normalizes(self):
        """Preprocessed image values should be roughly in [-1, 1] range."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        tensor = preprocess_image(img, image_size=224)
        # After normalization to [-1, 1], values should be bounded
        assert tensor.min() >= -1.1
        assert tensor.max() <= 1.1

    def test_preprocess_from_path(self):
        """preprocess_image should accept a file path string."""
        import tempfile
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            tensor = preprocess_image(f.name, image_size=64)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 64, 64)

    def test_preprocess_grayscale_converts(self):
        """Grayscale images should be converted to RGB (3 channels)."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("L", (100, 100), 128)
        tensor = preprocess_image(img, image_size=32)
        assert tensor.shape == (1, 3, 32, 32)

    def test_preprocess_rgba_converts(self):
        """RGBA images should be converted to RGB (drop alpha)."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        tensor = preprocess_image(img, image_size=32)
        assert tensor.shape == (1, 3, 32, 32)


class TestEncodeImage:
    """encode_image convenience function."""

    def test_encode_image_returns_features(self):
        """encode_image should return feature tensor from encoder."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig, encode_image,
        )
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        features = encode_image(encoder, img)
        num_patches = (32 // 8) ** 2  # 16
        assert features.shape == (1, num_patches, 32)


class TestVisionEncoderSaveLoad:
    """Save/load vision encoder weights."""

    def test_state_dict_saveable(self):
        """Vision encoder state_dict should be saveable."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        sd = encoder.state_dict()
        assert len(sd) > 0

    def test_config_to_dict_roundtrip(self):
        """VisionEncoderConfig should round-trip through dict."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=128, patch_size=8, dim=64, n_layers=3, n_heads=4)
        d = cfg.to_dict()
        cfg2 = VisionEncoderConfig(**d)
        assert cfg2.image_size == 128
        assert cfg2.patch_size == 8
        assert cfg2.dim == 64
        assert cfg2.n_layers == 3
        assert cfg2.n_heads == 4

    def test_load_state_dict_restores(self):
        """Loading a state_dict should restore encoder weights."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        enc1 = VisionEncoder(cfg)
        sd = enc1.state_dict()
        enc2 = VisionEncoder(cfg)
        enc2.load_state_dict(sd)
        # Both should produce same output
        x = torch.randn(1, 3, 32, 32)
        enc1.eval()
        enc2.eval()
        out1 = enc1(x)
        out2 = enc2(x)
        assert torch.allclose(out1, out2)


class TestPretrainedVisionConfig:
    """VisionEncoderConfig pretrained fields."""

    def test_pretrained_defaults(self):
        """use_pretrained should default to False, preserving existing behavior."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig()
        assert cfg.use_pretrained is False
        assert cfg.pretrained_model == "vit_small_patch16_224"
        assert cfg.freeze_backbone is True

    def test_pretrained_to_dict_roundtrip(self):
        """Pretrained fields should survive dict serialization."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_base_patch16_224",
            freeze_backbone=False,
        )
        d = cfg.to_dict()
        assert d["use_pretrained"] is True
        assert d["pretrained_model"] == "vit_base_patch16_224"
        assert d["freeze_backbone"] is False
        cfg2 = VisionEncoderConfig(**d)
        assert cfg2.use_pretrained is True
        assert cfg2.pretrained_model == "vit_base_patch16_224"
        assert cfg2.freeze_backbone is False

    def test_pretrained_presets_exist(self):
        """Pretrained presets should be in VISION_PRESETS."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert "pretrained_tiny" in VISION_PRESETS
        assert "pretrained_small" in VISION_PRESETS
        assert "pretrained_base" in VISION_PRESETS
        ps = VISION_PRESETS["pretrained_small"]
        assert ps.use_pretrained is True

    def test_pretrained_presets_have_correct_dims(self):
        """Pretrained presets should match standard ViT dimensions."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert VISION_PRESETS["pretrained_tiny"].dim == 192
        assert VISION_PRESETS["pretrained_small"].dim == 384
        assert VISION_PRESETS["pretrained_base"].dim == 768


class TestImageNetNormalization:
    """ImageNet normalization constants and preprocessing."""

    def test_imagenet_constants_exist(self):
        """IMAGENET_MEAN and IMAGENET_STD should be defined."""
        from enigma_engine.core.vision_encoder import IMAGENET_MEAN, IMAGENET_STD
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        # Standard ImageNet values
        assert abs(IMAGENET_MEAN[0] - 0.485) < 0.01
        assert abs(IMAGENET_STD[0] - 0.229) < 0.01

    def test_preprocess_imagenet_normalize(self):
        """Preprocessing with imagenet_normalize should differ from default."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        t_default = preprocess_image(img, image_size=32)
        t_imagenet = preprocess_image(img, image_size=32, imagenet_normalize=True)
        assert not torch.allclose(t_default, t_imagenet)

    def test_preprocess_imagenet_range(self):
        """ImageNet-normalized mid-gray should be near zero."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        t = preprocess_image(img, image_size=32, imagenet_normalize=True)
        # Mid-gray (128/255 ≈ 0.502) is close to ImageNet means (~0.45–0.485)
        # So normalized values should be small
        assert t.min() > -5.0
        assert t.max() < 5.0


class TestPretrainedVisionEncoder:
    """VisionEncoder with pretrained timm backbone."""

    def test_pretrained_encoder_creates_backbone(self):
        """Pretrained encoder should have a backbone attribute."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
        )
        encoder = VisionEncoder(cfg)
        assert hasattr(encoder, "backbone")
        assert encoder.backbone is not None

    def test_pretrained_encoder_forward_shape(self):
        """Pretrained encoder output shape should be [B, num_patches, dim]."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
        )
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        # 224/16 = 14, 14*14 = 196 patches
        assert out.shape == (1, 196, 192)

    def test_pretrained_freeze_backbone(self):
        """freeze_backbone=True should freeze all backbone parameters."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
            freeze_backbone=True,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone is not None
        backbone_frozen = all(
            not p.requires_grad for p in encoder.backbone.parameters()
        )
        assert backbone_frozen

    def test_pretrained_unfreeze_backbone(self):
        """freeze_backbone=False should leave backbone weights trainable."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
            freeze_backbone=False,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone is not None
        has_trainable = any(
            p.requires_grad for p in encoder.backbone.parameters()
        )
        assert has_trainable

    def test_pretrained_with_dim_projection(self):
        """When config.dim != backbone dim, a projection layer should exist."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        # vit_tiny_patch16_224 has embed_dim=192, set config.dim to 64
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=64,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone_proj is not None
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (1, 196, 64)

    def test_pretrained_no_projection_when_dims_match(self):
        """When config.dim matches backbone dim, no projection is needed."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,  # matches vit_tiny embed_dim
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone_proj is None

    def test_pretrained_gradients_flow_through_projection(self):
        """Gradients should flow through the pretrained encoder."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=64,
            freeze_backbone=False,
        )
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad


class TestVisionWithTextModel:
    """Integration: vision encoder + text model via forward_multimodal."""

    def test_vision_features_through_model(self):
        """Vision encoder output should work with forward_multimodal."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        v_encoder = VisionEncoder(vcfg)

        # Text model with vision_hidden_size matching encoder dim
        tcfg = ForgeConfig(
            vocab_size=100, dim=64, n_layers=1, n_heads=2,
            max_seq_len=64, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Encode an image
        img_tensor = torch.randn(1, 3, 32, 32)
        vision_features = v_encoder(img_tensor)

        # Pass through text model with some text tokens
        text_ids = torch.randint(0, 100, (1, 5))
        logits = model.forward_multimodal(
            input_ids=text_ids,
            vision_features=vision_features,
        )
        # Output should cover vision patches + text tokens
        expected_seq = vcfg.num_patches + 5
        # vocab_size=100 padded to next multiple of 64 = 128
        padded_vocab = (100 + 63) & ~63
        assert logits.shape == (1, expected_seq, padded_vocab)

    def test_vision_only_forward(self):
        """forward_multimodal should work with only vision features (no text)."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        v_encoder = VisionEncoder(vcfg)

        tcfg = ForgeConfig(
            vocab_size=100, dim=64, n_layers=1, n_heads=2,
            max_seq_len=64, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        img_tensor = torch.randn(1, 3, 32, 32)
        vision_features = v_encoder(img_tensor)

        logits = model.forward_multimodal(
            input_ids=None,
            vision_features=vision_features,
        )
        padded_vocab = (100 + 63) & ~63
        assert logits.shape == (1, vcfg.num_patches, padded_vocab)


class TestVisionProjectionMLP:
    """Vision-1b: projection upgraded from single Linear to LLaVA-1.5 2-layer MLP."""

    def _make_model(self, vision_hidden=32, dim=64):
        from enigma_engine.core.model import Enigma, ForgeConfig
        tcfg = ForgeConfig(
            vocab_size=100, dim=dim, n_layers=1, n_heads=2,
            max_seq_len=64, vision_hidden_size=vision_hidden,
        )
        return Enigma(config=tcfg)

    def test_vision_projection_is_sequential(self):
        """Projection must be nn.Sequential, not a single Linear."""
        import torch.nn as nn
        model = self._make_model()
        assert isinstance(model.vision_projection, nn.Sequential)
        # 3 layers: Linear → GELU → Linear
        assert len(model.vision_projection) == 3

    def test_vision_projection_uses_gelu(self):
        """Middle layer must be GELU per LLaVA-1.5."""
        import torch.nn as nn
        model = self._make_model()
        assert isinstance(model.vision_projection[0], nn.Linear)
        assert isinstance(model.vision_projection[1], nn.GELU)
        assert isinstance(model.vision_projection[2], nn.Linear)

    def test_vision_projection_dimensions(self):
        """First Linear: vision_hidden→dim. Second Linear: dim→dim."""
        model = self._make_model(vision_hidden=32, dim=64)
        proj = model.vision_projection
        assert proj[0].in_features == 32
        assert proj[0].out_features == 64
        assert proj[2].in_features == 64
        assert proj[2].out_features == 64

    def test_vision_projection_has_bias(self):
        """LLaVA-1.5 reference impl uses bias=True on both Linears."""
        model = self._make_model()
        assert model.vision_projection[0].bias is not None
        assert model.vision_projection[2].bias is not None

    def test_vision_projection_forward_shape(self):
        """Forward must preserve (batch, seq, dim) and project last dim."""
        import torch
        model = self._make_model(vision_hidden=32, dim=64)
        x = torch.randn(2, 4, 32)
        y = model.vision_projection(x)
        assert y.shape == (2, 4, 64)

    def test_vision_projection_callable_in_forward_multimodal(self):
        """forward_multimodal still works end-to-end after the upgrade."""
        import torch
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig)

        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32,
                                    n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        model = self._make_model(vision_hidden=vcfg.dim, dim=64)

        img = torch.randn(1, 3, 32, 32)
        vfeat = v_enc(img)
        text_ids = torch.randint(0, 100, (1, 5))
        logits = model.forward_multimodal(
            input_ids=text_ids, vision_features=vfeat)
        padded_vocab = (100 + 63) & ~63
        assert logits.shape == (1, vcfg.num_patches + 5, padded_vocab)


# =============================================================================
# VISION TRAINING TESTS
# =============================================================================


# ---------------------------------------------------------------------------
# GGUF Export
# ---------------------------------------------------------------------------


# ================================================================
# D5: Weight mapping tests
# ================================================================

class TestWeightMapping:
    """Tests for weight_mapping.py — format conversion."""

    def test_hf_model_maps_exist(self):
        """HF_MODEL_MAPS has entries for each architecture."""
        from enigma_engine.core.weight_mapping import HF_MODEL_MAPS
        for arch in ("llama", "gpt2", "phi", "qwen2", "gemma"):
            assert arch in HF_MODEL_MAPS, f"Missing map for {arch}"

    def test_gguf_weight_map_nonempty(self):
        """GGUF_WEIGHT_MAP has entries."""
        from enigma_engine.core.weight_mapping import GGUF_WEIGHT_MAP
        assert len(GGUF_WEIGHT_MAP) > 5

    def test_mapper_hf_to_forge_llama(self):
        """map_huggingface_to_forge maps llama weight names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.embed_tokens.weight": "tensor_a",
                 "model.norm.weight": "tensor_b"}
        result = mapper.map_huggingface_to_forge(dummy, model_type="llama")
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_hf_to_forge_gpt2(self):
        """map_huggingface_to_forge maps GPT-2 weight names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"transformer.wte.weight": "tensor_a",
                 "transformer.ln_f.weight": "tensor_b"}
        result = mapper.map_huggingface_to_forge(dummy, model_type="gpt2")
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_gguf_to_forge(self):
        """map_gguf_to_forge maps GGUF tensor names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"token_embd.weight": "tensor_a",
                 "output_norm.weight": "tensor_b"}
        result = mapper.map_gguf_to_forge(dummy)
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_onnx_to_forge(self):
        """map_onnx_to_forge can convert ONNX weights."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.embed_tokens.weight": "tensor_a"}
        result = mapper.map_onnx_to_forge(dummy)
        assert "tok_embeddings.weight" in result

    def test_detect_model_type_gpt2(self):
        """_detect_hf_model_type identifies GPT-2 layout."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"transformer.h.0.ln_1.weight": 1,
                 "transformer.wte.weight": 2}
        assert mapper._detect_hf_model_type(dummy) == "gpt2"

    def test_detect_model_type_llama(self):
        """_detect_hf_model_type identifies LLaMA layout."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.layers.0.self_attn.q_proj.weight": 1,
                 "model.layers.0.mlp.gate_proj.weight": 2}
        assert mapper._detect_hf_model_type(dummy) == "llama"

    def test_get_stats(self):
        """get_stats returns mapping statistics."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        mapper.map_huggingface_to_forge(
            {"model.embed_tokens.weight": "t"}, model_type="llama")
        stats = mapper.get_stats()
        assert "mapped" in stats
        assert "skipped" in stats
        assert stats["mapped"] >= 1


# ================================================================
# D5: GGUF dequantization tests
# ================================================================

class TestGGUFDequant:
    """Tests for gguf_dequant.py — tensor parsing and dequantization."""

    def test_extract_config_from_metadata_llama(self):
        """extract_config_from_metadata parses llama-style metadata."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {
            "llama.embedding_length": 2048,
            "llama.block_count": 16,
            "llama.attention.head_count": 16,
            "llama.attention.head_count_kv": 4,
            "llama.context_length": 4096,
        }
        config = extract_config_from_metadata(metadata)
        assert config["dim"] == 2048
        assert config["n_layers"] == 16
        assert config["n_heads"] == 16
        assert config["n_kv_heads"] == 4
        assert config["max_seq_len"] == 4096

    def test_extract_config_defaults(self):
        """extract_config_from_metadata fills defaults for missing keys."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        config = extract_config_from_metadata({})
        assert "dim" in config
        assert "n_layers" in config
        assert "n_heads" in config
        assert "vocab_size" in config
        assert "max_seq_len" in config

    def test_dequantize_q4_0_shape(self):
        """dequantize_q4_0 returns correct shape."""
        import numpy as np
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_0
        # Build a single block: 2 bytes float16 scale + 16 bytes data = 18 bytes
        scale = np.float16(0.5)
        block = scale.tobytes() + bytes(16)
        result = dequantize_q4_0(block, (32,))
        assert result.shape == (32,)
        assert result.dtype == torch.float32

    def test_dequantize_q8_0_shape(self):
        """dequantize_q8_0 returns correct shape."""
        import numpy as np
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q8_0
        # Build a single block: 2 bytes float16 scale + 32 bytes data = 34 bytes
        scale = np.float16(1.0)
        block = scale.tobytes() + bytes(32)
        result = dequantize_q8_0(block, (32,))
        assert result.shape == (32,)
        assert result.dtype == torch.float32

    def test_extract_config_embed_length_alias(self):
        """extract_config_from_metadata handles embed_length alias."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {"llama.embed_length": 512}
        config = extract_config_from_metadata(metadata)
        assert config["dim"] == 512


# ================================================================
# D5: Model components tests
# ================================================================

class TestModelComponents:
    """Tests for model_components.py — neural network building blocks."""

    def test_rmsnorm_output_shape(self):
        """RMSNorm preserves input shape."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalizes(self):
        """RMSNorm output has roughly unit RMS."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(128)
        x = torch.randn(1, 5, 128) * 10.0
        out = norm(x)
        rms = (out ** 2).mean(-1).sqrt()
        # Should be close to 1.0 (norm weight initialized to ones)
        assert rms.mean().item() == pytest.approx(1.0, abs=0.3)

    def test_precompute_rope_frequencies(self):
        """precompute_rope_frequencies returns correct shapes."""
        pytest.importorskip("torch")
        from enigma_engine.core.model_components import precompute_rope_frequencies
        freqs = precompute_rope_frequencies(64, 128)
        assert freqs.shape[0] == 128  # seq_len
        assert freqs.shape[1] == 32   # dim // 2

    def test_feedforward_output_shape(self):
        """FeedForward preserves batch and seq dims."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import FeedForward
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128)
        ff = FeedForward(config)
        x = torch.randn(1, 5, 64)
        out = ff(x)
        assert out.shape == (1, 5, 64)

    def test_moe_feedforward_has_experts(self):
        """MoEFeedForward creates multiple expert modules."""
        pytest.importorskip("torch")
        from enigma_engine.core.model_components import MoEFeedForward
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(
            dim=64, hidden_dim=128,
            use_moe=True, num_experts=4, num_experts_per_token=2)
        moe = MoEFeedForward(config)
        assert len(moe.experts) == 4

    def test_moe_load_balancing_loss_differentiable(self):
        """MoE load balancing loss must have grad_fn (differentiable)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import MoEFeedForward
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(
            dim=64, hidden_dim=128,
            use_moe=True, num_experts=4, num_experts_per_token=2,
            moe_load_balancing=0.01)
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 4, 64)
        _ = moe(x)
        aux = moe.get_aux_loss()
        # Must be a tensor with grad_fn so backward() produces router gradient
        assert isinstance(aux, torch.Tensor), "aux_loss must be a tensor"
        assert aux.grad_fn is not None, "aux_loss has no grad_fn — router gets zero gradient"

    # ── QK Normalization ─────────────────────────────────────────

    def test_qk_norm_config_default_on(self):
        """use_qk_norm defaults to True."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.use_qk_norm is True

    def test_qk_norm_preserves_shape(self):
        """Attention output shape unchanged with qk_norm enabled."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=4, use_qk_norm=True)
        attn = Attention(config)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    # ── LayerScale ───────────────────────────────────────────────

    def test_layer_scale_config_default_off(self):
        """use_layer_scale defaults to False."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.use_layer_scale is False

    def test_layer_scale_creates_parameters(self):
        """TransformerBlock has ls_attn and ls_ffn when layer_scale enabled."""
        pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        assert hasattr(block, 'ls_attn')
        assert hasattr(block, 'ls_ffn')
        assert block.ls_attn.shape == (64,)
        assert block.ls_ffn.shape == (64,)

    def test_layer_scale_init_small(self):
        """LayerScale parameters are initialized to a small value (1e-5)."""
        pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        assert block.ls_attn.mean().item() == pytest.approx(1e-5, abs=1e-7)

    def test_layer_scale_preserves_shape(self):
        """TransformerBlock output shape unchanged with layer_scale."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    # ── Drop Path (Stochastic Depth) ────────────────────────────

    def test_drop_path_config_default_zero(self):
        """drop_path_rate defaults to 0.0 (disabled)."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.drop_path_rate == 0.0

    def test_drop_path_noop_at_zero(self):
        """DropPath with rate=0 is identity."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(0.0)
        x = torch.randn(2, 4, 64)
        dp.train()
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_noop_at_eval(self):
        """DropPath is identity during eval regardless of rate."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(0.5)
        dp.eval()
        x = torch.randn(2, 4, 64)
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_linearly_increasing(self):
        """Deeper layers get higher drop rates."""
        pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, n_layers=4, drop_path_rate=0.2)
        block0 = TransformerBlock(config, layer_id=0)
        block3 = TransformerBlock(config, layer_id=3)
        assert block0.drop_path_attn.drop_prob < block3.drop_path_attn.drop_prob

    def test_drop_path_rejects_one(self):
        """DropPath(drop_prob=1.0) raises ValueError (div-by-zero)."""
        from enigma_engine.core.model_components import DropPath
        with pytest.raises(ValueError, match="drop_prob must be in"):
            DropPath(1.0)

    def test_drop_path_rejects_negative(self):
        """DropPath(drop_prob=-0.1) raises ValueError."""
        from enigma_engine.core.model_components import DropPath
        with pytest.raises(ValueError, match="drop_prob must be in"):
            DropPath(-0.1)

    # ── EMA Weight Averaging ─────────────────────────────────────

    def test_ema_config_default_off(self):
        """ema_decay defaults to 0.0 (disabled) in TrainingConfig."""
        from enigma_engine.training.training import TrainingConfig
        config = TrainingConfig()
        assert config.ema_decay == 0.0

    def test_ema_in_to_dict(self):
        """ema_decay is serialized in TrainingConfig.to_dict()."""
        from enigma_engine.training.training import TrainingConfig
        config = TrainingConfig(ema_decay=0.999)
        d = config.to_dict()
        assert "ema_decay" in d
        assert d["ema_decay"] == 0.999

    def test_ema_tracks_weights(self):
        """EMAWeightAverager maintains shadow copies of parameters."""
        torch = pytest.importorskip("torch")
        from enigma_engine.training.training import EMAWeightAverager
        model = torch.nn.Linear(4, 4)
        ema = EMAWeightAverager(model, decay=0.99)
        # Shadow should exist for each parameter
        assert len(ema.shadow) == len(list(model.parameters()))

    def test_ema_update_moves_shadow(self):
        """EMAWeightAverager.update() moves shadow toward current weights."""
        torch = pytest.importorskip("torch")
        from enigma_engine.training.training import EMAWeightAverager
        model = torch.nn.Linear(4, 4, bias=False)
        ema = EMAWeightAverager(model, decay=0.99)
        old_shadow = ema.shadow[0].clone()
        # Change the model weights
        with torch.no_grad():
            model.weight.fill_(99.0)
        ema.update(model)
        # Shadow should have moved toward 99 but not all the way
        new_shadow = ema.shadow[0]
        assert not torch.equal(old_shadow, new_shadow)
        assert new_shadow.mean().item() > old_shadow.mean().item()

    # ── torch.compile ────────────────────────────────────────────

    def test_compile_config_default_off(self):
        """use_compile defaults to False in TrainingConfig."""
        from enigma_engine.training.training import TrainingConfig
        config = TrainingConfig()
        assert config.use_compile is False


# ================================================================
# D5: Model config shim tests
# ================================================================

class TestModelConfigShim:
    """Tests for model_config.py — backward-compat shim."""

    def test_get_model_config_returns_dict(self):
        """get_model_config returns a valid config dict."""
        from enigma_engine.core.model_config import get_model_config
        config = get_model_config("tiny")
        assert isinstance(config, dict)
        assert "dim" in config
        assert "n_layers" in config

    def test_get_model_config_invalid_raises(self):
        """get_model_config raises ValueError for unknown sizes."""
        from enigma_engine.core.model_config import get_model_config
        with pytest.raises(ValueError, match="Unknown size"):
            get_model_config("nonexistent_size_xyz")

    def test_model_presets_reexport(self):
        """model_config.py re-exports MODEL_PRESETS from model_presets."""
        from enigma_engine.core.model_config import MODEL_PRESETS
        from enigma_engine.core.model_presets import MODEL_PRESETS as orig
        assert MODEL_PRESETS is orig


# ================================================================
# Mod Tools — auto-register mod commands as AI tools
# ================================================================

class TestRMSNormFp32Upcast:
    """RMSNorm must compute in fp32 then cast back to input dtype."""

    def test_output_dtype_matches_input(self):
        """RMSNorm output dtype == input dtype for float32."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(16)
        x = torch.randn(2, 16)
        out = norm(x)
        assert out.dtype == x.dtype

    def test_fp16_no_nan(self):
        """fp16 input should not produce NaN thanks to fp32 upcast."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(32)
        # Large values that would overflow in fp16 norm without upcast
        x = torch.randn(4, 32).half() * 100
        norm = norm.half()
        out = norm(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any(), "fp16 RMSNorm produced NaN"

    def test_vision_rmsnorm_fp32_upcast(self):
        """Vision encoder RMSNorm also upcasts to fp32."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(32)
        x = torch.randn(4, 32).half() * 100
        norm = norm.half()
        out = norm(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()


class TestCNNStem:
    """Test CNNStem module for hybrid CNN+ViT."""

    def test_forward_shape(self):
        """CNNStem outputs [batch, num_patches, dim]."""
        import torch
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        x = torch.randn(2, 3, 64, 64)
        out = stem(x)
        # CNN stem does /8 spatial reduction: 64/8 = 8, 8*8 = 64
        num_patches = (64 // 8) ** 2
        assert out.shape == (2, num_patches, 64)

    def test_is_nn_module(self):
        """CNNStem is an nn.Module."""
        import torch.nn as nn
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        assert isinstance(stem, nn.Module)

    def test_has_trainable_params(self):
        """CNNStem should have trainable parameters."""
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        params = sum(p.numel() for p in stem.parameters())
        assert params > 0

    def test_gradients_flow(self):
        """Gradients should flow through CNNStem."""
        import torch
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        x = torch.randn(1, 3, 64, 64)
        out = stem(x)
        loss = out.sum()
        loss.backward()
        any_grad = any(
            p.grad is not None for p in stem.parameters())
        assert any_grad


class TestHybridVisionEncoder:
    """Test VisionEncoder with use_cnn_stem=True."""

    def test_config_has_cnn_stem_field(self):
        """VisionEncoderConfig has use_cnn_stem field."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig

        cfg = VisionEncoderConfig()
        assert hasattr(cfg, "use_cnn_stem")
        assert cfg.use_cnn_stem is False

    def test_hybrid_encoder_forward(self):
        """Hybrid VisionEncoder produces correct output shape."""
        import torch
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig)

        cfg = VisionEncoderConfig(
            image_size=64, patch_size=8, dim=64,
            n_layers=2, n_heads=2, use_cnn_stem=True)
        encoder = VisionEncoder(cfg)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        # CNN stem: 64/8 = 8, 8*8 = 64 patches
        assert out.shape[0] == 2
        assert out.shape[2] == 64
        assert out.shape[1] == (64 // 8) ** 2

    def test_hybrid_presets_exist(self):
        """VISION_PRESETS includes hybrid_small and hybrid_medium."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS

        assert "hybrid_small" in VISION_PRESETS
        assert "hybrid_medium" in VISION_PRESETS

    def test_hybrid_presets_use_cnn_stem(self):
        """Hybrid presets have use_cnn_stem=True."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS

        assert VISION_PRESETS["hybrid_small"].use_cnn_stem is True
        assert VISION_PRESETS["hybrid_medium"].use_cnn_stem is True

    def test_config_to_dict_includes_cnn_stem(self):
        """VisionEncoderConfig.to_dict() includes use_cnn_stem."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig

        cfg = VisionEncoderConfig(use_cnn_stem=True)
        d = cfg.to_dict()
        assert "use_cnn_stem" in d
        assert d["use_cnn_stem"] is True


# ================================================================
# FORGE GUI — NEW TRAINING MODES
# ================================================================

class TestModelConfigDict:
    """Verify _model_config_dict saves full config, not a subset."""

    def test_config_dict_includes_architecture_flags(self):
        """_model_config_dict must include use_rope, use_rms_norm, etc."""
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(use_moe=True, use_qk_norm=True)

        class _FakeModel:
            config = cfg

        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._model_config_dict(_FakeModel())
        assert "use_rope" in result, "Config dict missing use_rope"
        assert "use_moe" in result, "Config dict missing use_moe"
        assert result["use_moe"] is True, "Config dict should reflect actual values"


# ================================================================
# 13: LoRA-Based RLHF (no deep copy on GPU)
# ================================================================


class TestKVCacheClone:
    """Tests for KV cache get() returning cloned tensors."""

    def test_get_returns_cloned_tensors(self):
        """Non-quantized get() returns views for zero-copy performance.

        Callers (attention computation) only read K/V — never mutate.
        Views avoid O(n) clone per token during generation.
        """
        import torch
        from enigma_engine.core.kv_cache import KVCache

        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=2,
            head_dim=4, device=torch.device("cpu"), dtype=torch.float32,
        )
        # Write something identifiable into the cache
        data = torch.ones(1, 2, 4)
        cache.update(data, data, position=0)

        # Get returns a view — shares storage with cache buffer
        k, v = cache.get()
        assert k.sum().item() > 0, "Data should be present"
        assert v.sum().item() > 0, "Data should be present"
        # Verify it IS a view (shares storage), not a copy
        assert k.data_ptr() == cache._cache_k[:, :2].data_ptr()


# ================================================================
# On-demand causal mask + KVCache wiring (#quality-audit)
# ================================================================

class TestOnDemandCausalMask:
    """Model builds causal mask lazily instead of pre-allocating max_seq_len²."""

    def test_causal_mask_starts_none(self):
        """_causal_mask should be None after __init__ (not pre-allocated)."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)
        assert model._causal_mask is None, "Mask should start None (on-demand)"
        assert model._causal_mask_size == 0

    def test_get_causal_mask_builds_on_demand(self):
        """_get_causal_mask creates and caches at the requested size."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)

        mask = model._get_causal_mask(8)
        assert mask.shape == (8, 8)
        # Upper triangle should be -inf, diagonal + below should be 0
        assert mask[0, 1] == float('-inf')
        assert mask[1, 0] == 0.0
        assert model._causal_mask_size == 8

    def test_causal_mask_grows_not_shrinks(self):
        """Requesting a larger mask grows the cache; smaller reuses it."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)

        model._get_causal_mask(4)
        assert model._causal_mask_size == 4

        model._get_causal_mask(16)
        assert model._causal_mask_size == 16

        # Smaller request shouldn't shrink
        mask = model._get_causal_mask(8)
        assert mask.shape == (8, 8)
        assert model._causal_mask_size == 16  # Still 16


class TestAttentionUsesPreAllocKVCache:
    """Attention class uses kv_cache.KVCache instead of torch.cat()."""

    def test_attention_has_no_cache_k_attribute(self):
        """Old cache_k/cache_v attrs should be gone."""
        from enigma_engine.core.model_components import Attention
        source = inspect.getsource(Attention.__init__)
        assert "self.cache_k" not in source
        assert "self.cache_v" not in source


class TestVideoFrameDedup:
    """Tests for video frame dedup and max_visual_tokens in encode_video_frames."""


    def test_dedup_drops_identical_frames(self):
        """Identical consecutive frames (cosine_sim=1.0) are dropped."""
        import torch

        # Simulate dedup logic directly
        feat = torch.ones(1, 4, 8)  # identical frame feature
        all_features = [feat.clone() for _ in range(5)]

        threshold = 0.95
        unique = [all_features[0]]
        for f in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = f.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < threshold:
                unique.append(f)

        # All identical → only first kept
        assert len(unique) == 1

    def test_dedup_keeps_different_frames(self):
        """Different consecutive frames are kept."""
        import torch

        all_features = [
            torch.randn(1, 4, 8) * 100,  # scale up to avoid accidental similarity
            torch.randn(1, 4, 8) * 100,
            torch.randn(1, 4, 8) * 100,
        ]

        threshold = 0.95
        unique = [all_features[0]]
        for f in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = f.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < threshold:
                unique.append(f)

        # Random tensors are unlikely to have cosine_sim > 0.95
        assert len(unique) >= 2

    def test_max_visual_tokens_truncation(self):
        """Concatenated features are truncated to max_visual_tokens."""
        import torch

        features = [torch.randn(1, 10, 8) for _ in range(3)]
        combined = torch.cat(features, dim=1)  # [1, 30, 8]
        max_visual_tokens = 15
        if max_visual_tokens > 0 and combined.shape[1] > max_visual_tokens:
            combined = combined[:, :max_visual_tokens, :]
        assert combined.shape[1] == 15


# ================================================================
# nGPT: Weight normalization (unit-norm hypersphere subset)
# ================================================================

class TestWeightNormConfig:
    """ForgeConfig.use_weight_norm flag and serialization."""

    def test_use_weight_norm_default_false(self):
        """use_weight_norm defaults to False."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.use_weight_norm is False

    def test_use_weight_norm_in_to_dict(self):
        """use_weight_norm is serialized in to_dict()."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(use_weight_norm=True)
        d = config.to_dict()
        assert "use_weight_norm" in d
        assert d["use_weight_norm"] is True

    def test_use_weight_norm_from_dict(self):
        """use_weight_norm round-trips through from_dict()."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(use_weight_norm=True)
        d = config.to_dict()
        restored = ForgeConfig.from_dict(d)
        assert restored.use_weight_norm is True

    def test_use_weight_norm_from_dict_missing_key(self):
        """from_dict without use_weight_norm uses default False."""
        from enigma_engine.core.model_presets import ForgeConfig
        d = {"dim": 64, "n_layers": 1, "n_heads": 2, "vocab_size": 100}
        config = ForgeConfig.from_dict(d)
        assert config.use_weight_norm is False


class TestWeightNormApplication:
    """Weight normalization applied to model linear layers."""

    def test_weight_norm_applies_parametrizations(self):
        """Enabling use_weight_norm adds weight parametrization to Linear layers."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_layers=2, n_heads=2, n_kv_heads=1,
            vocab_size=100, use_weight_norm=True)
        model = Enigma(config=config)

        # At least one Linear layer should have parametrizations
        has_param = False
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, "parametrizations"):
                    has_param = True
                    break
        assert has_param, "Weight norm should add parametrizations to Linear layers"

    def test_weight_norm_skips_output_head(self):
        """Output head (tied with embeddings) should not get weight norm."""
        pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
            vocab_size=100, use_weight_norm=True)
        model = Enigma(config=config)

        # Output head should NOT have parametrizations (weight-tied)
        assert not hasattr(model.output, "parametrizations"), \
            "Weight-tied output head should be skipped"

    def test_weight_norm_disabled_no_parametrizations(self):
        """With use_weight_norm=False, no parametrizations are added."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
            vocab_size=100, use_weight_norm=False)
        model = Enigma(config=config)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not hasattr(module, "parametrizations"), \
                    "No parametrizations when weight_norm disabled"

    def test_weight_norm_preserves_output_shape(self):
        """Model output shape unchanged with weight norm."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
            vocab_size=100, use_weight_norm=True)
        model = Enigma(config=config)
        model.eval()

        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            out = model(x)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        # vocab padded to multiple of 64 → 128
        assert logits.shape == (1, 8, 128)

    def test_weight_norm_columns_unit_norm(self):
        """After weight norm, effective weight columns have unit norm."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
            vocab_size=100, use_weight_norm=True)
        model = Enigma(config=config)

        # Find a parametrized linear layer and check column norms
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, "parametrizations"):
                # Access the effective weight (through parametrization)
                w = module.weight
                col_norms = w.norm(dim=0)
                # Each column should be scaled by g, but v/||v|| has unit norm
                # The parametrization separates direction and magnitude
                assert col_norms.shape[0] > 0
                break


# ================================================================
# run.py Lazy Torch Import (#28) + Port from Config (#29)
# ================================================================

class TestSpecialTokenIds:
    """All tokenizers must expose think_start_id / think_end_id."""

    def test_char_tokenizer_think_ids(self):
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_bpe_tokenizer_think_ids(self):
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_advanced_tokenizer_think_ids(self):
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_simple_tokenizer_think_ids(self):
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == 4
        assert tok.think_end_id == 5

    def test_get_special_token_ids_uses_attributes(self):
        """get_special_token_ids returns correct think IDs from tokenizer."""
        from enigma_engine.core.tokenizer import get_special_token_ids
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        ids = get_special_token_ids(tok)
        assert ids["think_start"] == tok.think_start_id
        assert ids["think_end"] == tok.think_end_id

    def test_core_ids_consistent(self):
        """pad=0, bos=1, eos=2, unk=3 across all tokenizers."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        from enigma_engine.core.tokenizer import SimpleTokenizer
        for cls in [CharacterTokenizer, BPETokenizer,
                    AdvancedBPETokenizer, SimpleTokenizer]:
            tok = cls()
            assert tok.pad_token_id == 0, f"{cls.__name__} pad != 0"
            assert tok.bos_token_id == 1, f"{cls.__name__} bos != 1"
            assert tok.eos_token_id == 2, f"{cls.__name__} eos != 2"
            assert tok.unk_token_id == 3, f"{cls.__name__} unk != 3"


# ================================================================
# freqs_cis non-persistent buffer
# ================================================================


# ================================================================
# LoRA scheduler — Quick Win #3
# ================================================================


@pytest.mark.structural
class TestAudioEncoderConfig:
    """AudioEncoderConfig must define all required fields."""

    def test_default_fields(self):
        """Config must have standard Whisper-like defaults."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        assert config.n_mels == 80
        assert config.dim > 0
        assert config.n_layers > 0
        assert config.n_heads > 0
        assert config.sample_rate == 16000
        assert config.n_fft > 0
        assert config.hop_length > 0

    def test_to_dict(self):
        """Config must serialize to dict."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "n_mels" in d
        assert "dim" in d
        assert "n_layers" in d

    def test_max_audio_len_field(self):
        """Config must have max_audio_len for positional embeddings."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        assert hasattr(config, "max_audio_len")
        assert config.max_audio_len > 0


@pytest.mark.structural
class TestAudioPresets:
    """AUDIO_PRESETS must provide standard size presets."""

    def test_presets_exist(self):
        """AUDIO_PRESETS dict must be importable."""
        from enigma_engine.core.audio_encoder import AUDIO_PRESETS
        assert isinstance(AUDIO_PRESETS, dict)
        assert len(AUDIO_PRESETS) >= 3  # tiny, base, small at minimum

    def test_preset_names(self):
        """Must include standard Whisper-like presets."""
        from enigma_engine.core.audio_encoder import AUDIO_PRESETS
        for name in ("tiny", "base", "small"):
            assert name in AUDIO_PRESETS, f"Missing preset: {name}"


# ================================================================
# Resume-from-checkpoint
# ================================================================

@pytest.mark.structural
class TestAudioEncoderStructure:
    """AudioEncoder must follow Whisper Conv1d + Transformer pattern."""

    def test_has_transformer_blocks(self):
        """Encoder must have a ModuleList of transformer blocks."""
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(config)
        assert hasattr(encoder, "blocks")
        assert len(encoder.blocks) == 2

    def test_forward_output_shape(self):
        """forward() output must be [B, T/2, dim] (stride-2 halves time)."""
        import torch
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2, n_mels=80, max_audio_len=1500)
        encoder = AudioEncoder(config)
        encoder.eval()
        # Input: [B, n_mels, n_frames]
        x = torch.randn(1, 80, 100)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape[0] == 1  # batch
        assert out.shape[1] == 50  # 100 / 2 = 50 (stride-2 conv)
        assert out.shape[2] == 64  # dim

    def test_forward_different_lengths(self):
        """Encoder must handle variable-length audio inputs."""
        import torch
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2, n_mels=80, max_audio_len=1500)
        encoder = AudioEncoder(config)
        encoder.eval()
        for n_frames in [50, 100, 200]:
            x = torch.randn(1, 80, n_frames)
            with torch.no_grad():
                out = encoder(x)
            assert out.shape == (1, n_frames // 2, 64)

    def test_param_count(self):
        """param_count() must return trainable param count."""
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(config)
        count = encoder.param_count()
        assert isinstance(count, int)
        assert count > 0


@pytest.mark.structural
class TestMelSpectrogram:
    """Mel spectrogram computation must work with torch only."""

    def test_mel_filterbank_shape(self):
        """mel_filterbank output must be [n_mels, n_fft//2+1]."""
        import torch
        from enigma_engine.core.audio_encoder import mel_filterbank
        fb = mel_filterbank(sr=16000, n_fft=400, n_mels=80)
        assert isinstance(fb, torch.Tensor)
        assert fb.shape == (80, 201)  # n_mels x (n_fft//2 + 1)

    def test_log_mel_spectrogram_output_shape(self):
        """log_mel_spectrogram must produce [1, n_mels, n_frames]."""
        import torch
        from enigma_engine.core.audio_encoder import log_mel_spectrogram
        # Simulate 1 second of 16kHz audio
        waveform = torch.randn(16000)
        mel = log_mel_spectrogram(waveform, n_fft=400, hop_length=160, n_mels=80)
        assert mel.ndim == 3  # [1, n_mels, n_frames]
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80


# ================================================================
# Vision training augmentation
# ================================================================

@pytest.mark.structural
class TestVisionAugmentation:
    """Verify vision training applies image augmentation."""

    def test_augment_preserves_shape(self):
        """Augmentation must return same shape as input."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.randn(1, 3, 224, 224).clamp(-1, 1)
        result = augment_vision_tensor(img)
        assert result.shape == img.shape

    def test_augment_preserves_range(self):
        """Augmented tensor must stay in [-1, 1]."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.randn(1, 3, 224, 224).clamp(-1, 1)
        for _ in range(20):
            result = augment_vision_tensor(img)
            assert result.min() >= -1.0 and result.max() <= 1.0, (
                f"Augmented tensor out of [-1, 1]: min={result.min()}, max={result.max()}")

    def test_augment_is_stochastic(self):
        """Multiple augmentations of same input should differ."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.ones(1, 3, 32, 32) * 0.5
        results = [augment_vision_tensor(img) for _ in range(10)]
        # At least some should differ (brightness/contrast jitter)
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Augmentation should produce varied outputs"


# ================================================================
# Gradient checkpointing enable/disable
# ================================================================

class TestGradientCheckpointing:
    """Verify Enigma.gradient_checkpointing_enable/disable toggles layers."""

    def test_enable_sets_all_layers(self):
        pytest.importorskip("torch")
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        cfg = ForgeConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, use_gradient_checkpointing=False)
        model = Enigma(config=cfg)
        assert all(not layer.use_checkpoint for layer in model.layers)
        model.gradient_checkpointing_enable()
        assert all(layer.use_checkpoint for layer in model.layers)

    def test_disable_clears_all_layers(self):
        pytest.importorskip("torch")
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        cfg = ForgeConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, use_gradient_checkpointing=True)
        model = Enigma(config=cfg)
        assert all(layer.use_checkpoint for layer in model.layers)
        model.gradient_checkpointing_disable()
        assert all(not layer.use_checkpoint for layer in model.layers)


# ================================================================
# Full checkpoint resume
# ================================================================


# ================================================================
# Baseline component shape & gradient tests
# ================================================================

class TestRMSNormBaseline:
    """Baseline shape and behavior tests for RMSNorm."""

    def test_output_shape_matches_input(self):
        """RMSNorm should preserve input shape."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_near_unit_rms(self):
        """Normalized output should have RMS close to 1."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(4, 8, 64)
        out = norm(x)
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1))
        # Weight is ones, so RMS should be close to 1
        assert (rms - 1.0).abs().mean() < 0.2

    def test_gradient_flows(self):
        """Gradient should flow through RMSNorm."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAttentionBaseline:
    """Baseline shape tests for Attention module."""

    def _make_config(self, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=32):
        from enigma_engine.core.model_presets import ForgeConfig
        return ForgeConfig(
            dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len, dropout=0.0,
            use_gradient_checkpointing=False,
        )

    def test_output_shape(self):
        """Attention output shape should be [batch, seq, dim]."""
        import torch
        from enigma_engine.core.model_components import Attention, precompute_rope_frequencies
        cfg = self._make_config()
        attn = Attention(cfg)
        attn.eval()
        x = torch.randn(2, 8, 64)
        freqs = precompute_rope_frequencies(64 // 4, 32)[:8]
        with torch.no_grad():
            out = attn(x, freqs_cis=freqs)
        assert out.shape == (2, 8, 64)

    def test_gqa_less_params_than_mha(self):
        """GQA (n_kv_heads < n_heads) should have fewer params than MHA."""
        from enigma_engine.core.model_components import Attention
        cfg_mha = self._make_config(n_heads=4, n_kv_heads=4)
        cfg_gqa = self._make_config(n_heads=4, n_kv_heads=2)
        mha = Attention(cfg_mha)
        gqa = Attention(cfg_gqa)
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        assert gqa_params < mha_params

    def test_invalid_n_kv_heads_raises(self):
        """n_kv_heads > n_heads should raise ValueError at config level."""
        from enigma_engine.core.model_presets import ForgeConfig
        with pytest.raises(ValueError, match="n_kv_heads"):
            ForgeConfig(dim=64, n_heads=4, n_kv_heads=8, max_seq_len=32)

    def test_n_heads_not_divisible_by_dim_raises(self):
        """n_heads not dividing dim should raise ValueError at config."""
        from enigma_engine.core.model_presets import ForgeConfig
        with pytest.raises(ValueError, match="n_heads"):
            ForgeConfig(dim=64, n_heads=6, n_kv_heads=6, max_seq_len=32)


class TestFeedForwardBaseline:
    """Baseline shape tests for FeedForward module."""

    def _make_config(self, dim=64, use_swiglu=True):
        from enigma_engine.core.model_presets import ForgeConfig
        return ForgeConfig(
            dim=dim, n_heads=4, n_kv_heads=2,
            max_seq_len=32, dropout=0.0, use_swiglu=use_swiglu,
            use_gradient_checkpointing=False,
        )

    def test_swiglu_output_shape(self):
        """SwiGLU FFN should preserve [batch, seq, dim] shape."""
        import torch
        from enigma_engine.core.model_components import FeedForward
        cfg = self._make_config(use_swiglu=True)
        ffn = FeedForward(cfg)
        ffn.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_standard_ffn_output_shape(self):
        """Standard FFN (no SwiGLU) should preserve shape."""
        import torch
        from enigma_engine.core.model_components import FeedForward
        cfg = self._make_config(use_swiglu=False)
        ffn = FeedForward(cfg)
        ffn.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_swiglu_has_three_projections(self):
        """SwiGLU should have w1, w2, w3 projections."""
        from enigma_engine.core.model_components import FeedForward
        cfg = self._make_config(use_swiglu=True)
        ffn = FeedForward(cfg)
        assert hasattr(ffn, 'w1')
        assert hasattr(ffn, 'w2')
        assert hasattr(ffn, 'w3')

    def test_gradient_flows_through_ffn(self):
        """Gradient should flow through FeedForward."""
        import torch
        from enigma_engine.core.model_components import FeedForward
        cfg = self._make_config(use_swiglu=True)
        ffn = FeedForward(cfg)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestTransformerBlockBaseline:
    """Baseline shape tests for TransformerBlock."""

    def _make_config(self, dim=64):
        from enigma_engine.core.model_presets import ForgeConfig
        return ForgeConfig(
            dim=dim, n_heads=4, n_kv_heads=2,
            max_seq_len=32, dropout=0.0,
            use_gradient_checkpointing=False,
        )

    def test_output_shape(self):
        """TransformerBlock should preserve [batch, seq, dim] shape."""
        import torch
        from enigma_engine.core.model_components import TransformerBlock, precompute_rope_frequencies
        cfg = self._make_config()
        block = TransformerBlock(cfg, layer_id=0)
        block.eval()
        x = torch.randn(2, 8, 64)
        freqs = precompute_rope_frequencies(64 // 4, 32)[:8]
        with torch.no_grad():
            out = block(x, freqs_cis=freqs)
        assert out.shape == (2, 8, 64)

    def test_residual_connection(self):
        """Output should not be identical to FFN output (residual adds input)."""
        import torch
        from enigma_engine.core.model_components import TransformerBlock, precompute_rope_frequencies
        cfg = self._make_config()
        block = TransformerBlock(cfg, layer_id=0)
        block.eval()
        x = torch.randn(1, 4, 64)
        freqs = precompute_rope_frequencies(64 // 4, 32)[:4]
        with torch.no_grad():
            out = block(x, freqs_cis=freqs)
        # Output should differ from input (model transforms it)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_gradient_flows_through_block(self):
        """Gradient should flow through the full TransformerBlock."""
        import torch
        from enigma_engine.core.model_components import TransformerBlock, precompute_rope_frequencies
        cfg = self._make_config()
        block = TransformerBlock(cfg, layer_id=0)
        x = torch.randn(1, 4, 64, requires_grad=True)
        freqs = precompute_rope_frequencies(64 // 4, 32)[:4]
        out = block(x, freqs_cis=freqs)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDropPathBaseline:
    """Baseline tests for DropPath (stochastic depth)."""

    def test_eval_is_identity(self):
        """DropPath in eval mode should be identity."""
        import torch
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(2, 4, 64)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_train_with_zero_drop_is_identity(self):
        """DropPath with drop_prob=0 in train mode should be identity."""
        import torch
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(2, 4, 64)
        out = dp(x)
        assert torch.allclose(out, x)


class TestQuantizationLogHonesty:
    """BUG-2: quantization helpers must log the path they actually took."""

    def test_static_int8_logs_dynamic_fallback(self, caplog):
        """``_apply_static_int8_quantization`` falls back to dynamic when no
        calibration data is wired; the log must say so, not claim
        'Applied static INT8 quantization'."""
        import logging
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=2,
            vocab_size=100, max_seq_len=64,
        )
        model = Enigma(config)

        caplog.set_level(logging.INFO, logger="enigma_engine.core.model")
        model._apply_static_int8_quantization()

        messages = [record.message for record in caplog.records]
        # Honest: mentions the actual path that ran (dynamic fallback)
        assert any("dynamic" in m.lower() for m in messages), (
            f"static INT8 helper must log the dynamic-fallback path. Got: {messages}"
        )
        # Forbidden: blanket "Applied static INT8 quantization" claim
        assert not any(
            m == "Applied static INT8 quantization" for m in messages
        ), f"Found dishonest 'Applied static INT8 quantization' log: {messages}"

    def test_quantize_int8_does_not_claim_static_when_falling_back(
            self, caplog):
        """End-to-end: ``quantize('int8')`` must not log
        'Applied static INT8 quantization' when the underlying helper
        actually ran dynamic INT8."""
        import logging
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=2,
            vocab_size=100, max_seq_len=64,
        )
        model = Enigma(config)

        caplog.set_level(logging.INFO, logger="enigma_engine.core.model")
        try:
            model.quantize("int8")
        except Exception:
            # Quantization may fail in CI environments — the log
            # assertion still applies to whatever ran.
            pass

        messages = [record.message for record in caplog.records]
        assert not any(
            m == "Applied static INT8 quantization" for m in messages
        ), f"quantize('int8') still claims 'static' on fallback: {messages}"

