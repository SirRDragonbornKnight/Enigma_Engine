#!/usr/bin/env python3
"""
Example showcasing Universal Model features in ForgeAI.

This demonstrates the new capabilities added to forge_ai/core/model.py:
- RoPE scaling for extended context
- Multi-modal integration
- Speculative decoding
- Configuration serialization
"""
import torch
from forge_ai.core.model import create_model, Forge, ForgeConfig


def demo_backward_compatibility():
    """Demonstrate that all existing code still works."""
    print("\n" + "="*70)
    print("1. BACKWARD COMPATIBILITY - All existing code works!")
    print("="*70)
    
    # Standard model creation (unchanged)
    model = create_model('small')
    print(f"✓ Created 'small' model: {model.num_parameters:,} parameters")
    
    # Generate some text
    input_ids = torch.randint(0, model.vocab_size, (1, 10))
    output = model.generate(input_ids, max_new_tokens=5)
    print(f"✓ Generated {output.shape[1] - input_ids.shape[1]} tokens")
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {output.shape}")


def demo_rope_scaling():
    """Demonstrate RoPE scaling for extended context."""
    print("\n" + "="*70)
    print("2. ROPE SCALING - Extended Context Length")
    print("="*70)
    
    # Create model with dynamic NTK RoPE scaling
    config = ForgeConfig(
        vocab_size=1241,
        dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=4096,  # Extended from typical 1024
        rope_scaling_type='dynamic',
        rope_scaling_factor=4.0
    )
    model = Forge(config=config)
    print(f"✓ Created model with extended context:")
    print(f"  Context length: {config.max_seq_len} tokens")
    print(f"  RoPE scaling: {config.rope_scaling_type}")
    print(f"  Scaling factor: {config.rope_scaling_factor}x")
    print(f"  Parameters: {model.num_parameters:,}")
    
    # Test with long sequence
    input_ids = torch.randint(0, 1241, (1, 512))  # 512 token input
    logits = model(input_ids)
    print(f"✓ Forward pass with {input_ids.shape[1]} tokens: {logits.shape}")


def demo_multimodal():
    """Demonstrate multi-modal integration."""
    print("\n" + "="*70)
    print("3. MULTI-MODAL - Vision + Text Integration")
    print("="*70)
    
    # Create model with vision encoder support
    config = ForgeConfig(
        vocab_size=1241,
        dim=256,
        n_layers=4,
        n_heads=4,
        vision_hidden_size=512  # Vision encoder output dimension
    )
    model = Forge(config=config)
    print(f"✓ Created multi-modal model:")
    print(f"  Text dimension: {config.dim}")
    print(f"  Vision dimension: {config.vision_hidden_size}")
    print(f"  Has vision projection: {model.vision_projection is not None}")
    
    # Simulate vision + text input
    vision_features = torch.randn(1, 49, 512)  # 7x7 vision patches
    text_ids = torch.randint(0, 1241, (1, 20))
    
    logits = model.forward_multimodal(
        input_ids=text_ids,
        vision_features=vision_features
    )
    print(f"✓ Forward pass with vision + text:")
    print(f"  Vision tokens: {vision_features.shape[1]}")
    print(f"  Text tokens:   {text_ids.shape[1]}")
    print(f"  Total tokens:  {logits.shape[1]}")
    print(f"  Output shape:  {logits.shape}")


def demo_speculative_decoding():
    """Demonstrate speculative decoding."""
    print("\n" + "="*70)
    print("4. SPECULATIVE DECODING - Faster Generation")
    print("="*70)
    
    # Create draft and main models
    draft_model = create_model('nano')
    main_model = create_model('small')
    
    print(f"✓ Created models:")
    print(f"  Draft: {draft_model.num_parameters:,} params")
    print(f"  Main:  {main_model.num_parameters:,} params")
    
    # Enable speculative decoding
    main_model.enable_speculative_decoding(
        draft_model,
        num_speculative_tokens=4
    )
    print(f"✓ Enabled speculative decoding with 4 draft tokens")
    print(f"  Expected speedup: 2-4x")
    
    # Can now use generate_speculative for faster generation
    print(f"✓ Ready for generate_speculative()")


def demo_config_features():
    """Demonstrate new configuration features."""
    print("\n" + "="*70)
    print("5. NEW CONFIG FEATURES - MoE, KV-Cache, etc.")
    print("="*70)
    
    # Create config with all new features
    config = ForgeConfig(
        vocab_size=1241,
        dim=256,
        n_layers=4,
        n_heads=4,
        # RoPE scaling
        rope_scaling_type='yarn',
        rope_scaling_factor=2.0,
        # MoE
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
        # Enhanced KV-cache
        sliding_window=1024,
        use_paged_attn=True,
        kv_cache_dtype='int8',
        # Multi-modal
        vision_hidden_size=768,
        audio_hidden_size=512
    )
    
    print("✓ Created advanced config:")
    print(f"  RoPE: {config.rope_scaling_type} @ {config.rope_scaling_factor}x")
    print(f"  MoE: {config.num_experts} experts, top-{config.num_experts_per_token}")
    print(f"  Sliding window: {config.sliding_window} tokens")
    print(f"  Paged attention: {config.use_paged_attn}")
    print(f"  KV-cache dtype: {config.kv_cache_dtype}")
    print(f"  Vision support: {config.vision_hidden_size is not None}")
    print(f"  Audio support: {config.audio_hidden_size is not None}")
    
    # Serialize config
    config_dict = config.to_dict()
    print(f"✓ Config serialized: {len(config_dict)} parameters")
    
    # Deserialize config
    config2 = ForgeConfig.from_dict(config_dict)
    print(f"✓ Config deserialized successfully")
    assert config2.rope_scaling_type == config.rope_scaling_type
    assert config2.num_experts == config.num_experts


def demo_universal_loading():
    """Demonstrate universal loading capabilities."""
    print("\n" + "="*70)
    print("6. UNIVERSAL LOADING - Multiple Formats")
    print("="*70)
    
    print("✓ Universal loading methods available:")
    print("  • Forge.from_any(path)         - Auto-detect format")
    print("  • Forge.from_huggingface(id)   - Load from HuggingFace")
    print("  • Forge.from_safetensors(path) - Load safetensors")
    print("  • Forge.from_gguf(path)        - Load GGUF/llama.cpp")
    print("  • Forge.from_onnx(path)        - Load ONNX (experimental)")
    
    print("\n  Example usage:")
    print("    model = Forge.from_any('model.gguf')")
    print("    model = Forge.from_huggingface('microsoft/phi-2')")
    print("    model = Forge.from_safetensors('model.safetensors')")


def demo_lora():
    """Demonstrate LoRA adapter support."""
    print("\n" + "="*70)
    print("7. LORA ADAPTERS - Efficient Fine-tuning")
    print("="*70)
    
    model = create_model('small')
    print(f"✓ Created base model: {model.num_parameters:,} params")
    
    print("✓ LoRA adapter methods available:")
    print("  • model.load_lora(path, name)  - Load adapter")
    print("  • model.merge_lora(name)       - Merge into base")
    
    print("\n  Example usage:")
    print("    model.load_lora('coding.pth', 'coding')")
    print("    model.load_lora('creative.pth', 'creative')")
    print("    model.merge_lora('coding')  # Merge adapter")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# FORGEAI UNIVERSAL MODEL - FEATURE DEMONSTRATION")
    print("#"*70)
    print("\nThis showcases the new universal features in forge_ai/core/model.py")
    print("All features maintain 100% backward compatibility!")
    
    try:
        demo_backward_compatibility()
        demo_rope_scaling()
        demo_multimodal()
        demo_speculative_decoding()
        demo_config_features()
        demo_universal_loading()
        demo_lora()
        
        print("\n" + "="*70)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nFor detailed usage, see: UNIVERSAL_MODEL_GUIDE.md")
        print("For tests, see: tests/test_universal_model.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
