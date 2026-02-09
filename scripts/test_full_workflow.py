#!/usr/bin/env python3
"""
Test Full Workflow (Dry Run)

Tests the entire API Training workflow WITHOUT making real API calls:
1. Secure key storage
2. API configuration
3. Training data generation (mock)
4. Model training
5. Model inference

Run: python scripts/test_full_workflow.py
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_step(step: int, msg: str):
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {msg}")
    print('='*60)


def print_result(success: bool, msg: str):
    """Print a result."""
    status = "PASS" if success else "FAIL"
    symbol = "[+]" if success else "[-]"
    print(f"  {symbol} {status}: {msg}")
    return success


def test_secure_key_storage():
    """Test 1: Secure key storage and retrieval."""
    print_step(1, "Secure Key Storage")
    
    from enigma_engine.utils.api_key_encryption import SecureKeyStorage
    
    # Use temp directory to avoid polluting real storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = SecureKeyStorage(storage_dir=Path(temp_dir))
        
        # Store a test key
        test_key = "sk-test-fake-key-12345"
        test_provider = "openai"
        
        result1 = storage.store_key(test_provider, test_key, "Test key for workflow")
        print_result(result1, "Store key")
        
        # Retrieve the key
        retrieved = storage.get_key(test_provider)
        result2 = retrieved == test_key
        print_result(result2, f"Retrieve key (match: {retrieved == test_key})")
        
        # List stored keys
        keys = storage.list_keys()
        result3 = test_provider in keys
        print_result(result3, f"List keys: {keys}")
        
        # Delete key
        result4 = storage.delete_key(test_provider)
        print_result(result4, "Delete key")
        
        # Verify deletion
        result5 = storage.get_key(test_provider) is None
        print_result(result5, "Verify deletion")
        
    return all([result1, result2, result3, result4, result5])


def test_api_configuration():
    """Test 2: API provider configuration."""
    print_step(2, "API Configuration")
    
    from enigma_engine.core.trainer_ai import get_trainer_ai, get_api_training_provider
    
    trainer = get_trainer_ai()
    api = get_api_training_provider()
    
    # Test configuring OpenAI (with fake key, won't make real calls)
    try:
        trainer.configure_api("openai", "sk-fake-test-key")
        result1 = True
    except Exception as e:
        result1 = False
        print(f"       Error: {e}")
    print_result(result1, "Configure OpenAI provider")
    
    # Check provider is set on the singleton (access internal attributes for testing)
    result2 = "openai" in api._providers
    print_result(result2, f"OpenAI provider registered: {'openai' in api._providers}")
    
    # Test configuring Anthropic
    try:
        trainer.configure_api("anthropic", "sk-ant-fake-test-key")
        result3 = True
    except Exception as e:
        result3 = False
        print(f"       Error: {e}")
    print_result(result3, "Configure Anthropic provider")
    
    # Check Anthropic is registered
    result4 = "anthropic" in api._providers
    print_result(result4, f"Anthropic provider registered: {'anthropic' in api._providers}")
    
    return all([result1, result2, result3, result4])


def test_mock_data_generation():
    """Test 3: Generate mock training data (no API calls)."""
    print_step(3, "Mock Data Generation")
    
    from enigma_engine.core.trainer_ai import TRAINING_TASKS
    
    # Create mock training data manually
    mock_data = []
    tasks_to_test = ["chat", "code", "avatar"]
    
    for task in tasks_to_test:
        task_info = TRAINING_TASKS.get(task, {})
        desc = task_info.get("description", "Unknown task")
        
        # Generate 5 mock examples per task
        for i in range(5):
            mock_data.append({
                "input": f"[Test {task.upper()} input {i+1}] User query about {desc}",
                "output": f"[Test {task.upper()} output {i+1}] AI response demonstrating {desc}",
                "task": task
            })
        print_result(True, f"Generated 5 mock examples for '{task}'")
    
    result = len(mock_data) == 15
    print_result(result, f"Total mock examples: {len(mock_data)}")
    
    return result, mock_data


def test_model_training(mock_data: list):
    """Test 4: Train a model on mock data."""
    print_step(4, "Model Training")
    
    from enigma_engine.core.model import create_model
    from enigma_engine.core.training import Trainer, TrainingConfig
    from enigma_engine.core.tokenizer import get_tokenizer
    
    # Create nano model for fast testing
    model = create_model("nano")
    tokenizer = get_tokenizer()
    
    # Get the model's config for later
    model_config = {
        "max_seq_len": model.config.max_seq_len,
        "vocab_size": model.config.vocab_size
    }
    
    result1 = model is not None
    print_result(result1, f"Created nano model")
    
    # Prepare training texts from mock data (as list of Q&A pairs)
    training_texts = []
    for item in mock_data:
        training_texts.append(f"Q: {item['input']}\nA: {item['output']}")
    
    result2 = len(training_texts) > 0
    print_result(result2, f"Prepared {len(training_texts)} training examples")
    
    # Train for 1 epoch
    config = TrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=0.0001,
        max_seq_len=model_config["max_seq_len"]
    )
    
    trainer = Trainer(model, tokenizer, config)
    
    try:
        train_result = trainer.train(training_texts)
        result3 = train_result is not None
        loss = train_result.get('final_loss', 0)
        print_result(result3, f"Training completed (loss: {loss:.4f})")
    except Exception as e:
        result3 = False
        print_result(result3, f"Training failed: {e}")
    
    # Save model to temp location
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_model_path = f.name
    
    try:
        import torch
        torch.save({
            "state_dict": model.state_dict(),
            "config": model_config,
            "tasks": list(set(item["task"] for item in mock_data))
        }, temp_model_path)
        result4 = os.path.exists(temp_model_path)
        print_result(result4, f"Saved model to temp file")
    except Exception as e:
        result4 = False
        print_result(result4, f"Save failed: {e}")
    
    return all([result1, result2, result3, result4]), temp_model_path, model_config


def test_model_inference(model_path: str, model_config: dict):
    """Test 5: Load and run inference on trained model."""
    print_step(5, "Model Inference")
    
    import torch
    from enigma_engine.core.model import create_model
    from enigma_engine.core.inference import EnigmaEngine
    from enigma_engine.core.tokenizer import get_tokenizer
    
    # Load saved model
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        result1 = "state_dict" in checkpoint
        print_result(result1, "Loaded checkpoint")
    except Exception as e:
        print_result(False, f"Failed to load checkpoint: {e}")
        return False
    
    # Recreate nano model and load weights
    model = create_model("nano")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    result2 = True
    print_result(result2, "Model weights loaded")
    
    # Create inference engine using from_model classmethod
    tokenizer = get_tokenizer()
    engine = EnigmaEngine.from_model(model, tokenizer)
    
    result3 = engine is not None
    print_result(result3, "Inference engine created")
    
    # Generate a test response
    test_prompt = "Hello, how are you?"
    try:
        response = engine.generate(test_prompt, max_gen=20)
        result4 = len(response) > 0
        print_result(result4, f"Generated response ({len(response)} chars)")
        print(f"       Prompt: '{test_prompt}'")
        print(f"       Response: '{response[:100]}...'") if len(response) > 100 else print(f"       Response: '{response}'")
    except Exception as e:
        result4 = False
        print_result(result4, f"Generation failed: {e}")
    
    # Clean up temp file
    try:
        os.remove(model_path)
        result5 = True
        print_result(result5, "Cleaned up temp model file")
    except Exception as e:
        result5 = False
        print_result(result5, f"Cleanup failed: {e}")
    
    return all([result1, result2, result3, result4, result5])


def test_gui_components():
    """Test 6: Verify GUI components exist (no display)."""
    print_step(6, "GUI Components (Import Check)")
    
    try:
        from enigma_engine.gui.tabs.build_ai_tab import create_build_ai_tab
        result1 = True
        print_result(result1, "create_build_ai_tab imported")
    except ImportError as e:
        result1 = False
        print_result(result1, f"create_build_ai_tab import failed: {e}")
    
    try:
        from enigma_engine.gui.tabs.build_ai_tab import APIDataGenWorker
        result2 = True
        print_result(result2, "APIDataGenWorker imported")
    except ImportError as e:
        result2 = False
        print_result(result2, f"APIDataGenWorker import failed: {e}")
    
    # Check TRAINING_TASKS is available
    try:
        from enigma_engine.core.trainer_ai import TRAINING_TASKS
        result3 = len(TRAINING_TASKS) == 12
        print_result(result3, f"TRAINING_TASKS has {len(TRAINING_TASKS)} tasks")
    except Exception as e:
        result3 = False
        print_result(result3, f"TRAINING_TASKS check failed: {e}")
    
    return all([result1, result2, result3])


def main():
    """Run full workflow test."""
    print("\n" + "="*60)
    print("ENIGMA AI ENGINE - FULL WORKFLOW TEST (DRY RUN)")
    print("="*60)
    print("\nThis test validates the entire API training workflow")
    print("WITHOUT making actual API calls.\n")
    
    results = {}
    
    # Test 1: Secure key storage
    results["secure_storage"] = test_secure_key_storage()
    
    # Test 2: API configuration
    results["api_config"] = test_api_configuration()
    
    # Test 3: Mock data generation
    success, mock_data = test_mock_data_generation()
    results["data_gen"] = success
    
    # Test 4: Model training
    success, model_path, model_config = test_model_training(mock_data)
    results["training"] = success
    
    # Test 5: Model inference
    results["inference"] = test_model_inference(model_path, model_config)
    
    # Test 6: GUI components
    results["gui"] = test_gui_components()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[-]"
        print(f"  {symbol} {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nThe full workflow is ready. To use with real API:")
        print("  1. Run: python run.py --gui")
        print("  2. Go to Build AI > Step 5: Training Data")
        print("  3. Select API provider and enter your key")
        print("  4. Choose tasks and generate training data")
        print("  5. Train your model on the generated data")
    else:
        print("SOME TESTS FAILED - Review output above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
