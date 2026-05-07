"""
Quick validation script to test the evaluation framework setup.
Run this after setting up your .env file to verify everything works.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        from src import dataset_loader, endpoint_client, evaluator, metrics, prompt_builder, runner, utils
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from config import config
        
        issues = config.validate()
        if issues:
            print("⚠ Configuration warnings:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ Configuration loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_dataset_loader():
    """Test dataset loading."""
    print("\nTesting dataset loader...")
    try:
        from src.dataset_loader import DatasetLoader
        
        loader = DatasetLoader("writing_assistant", version="1.2", split="test")
        
        # Try to load (will use cache if available)
        test_cases = loader.load()
        
        if test_cases:
            print(f"✓ Loaded {len(test_cases)} test cases")
            print(f"  First test case ID: {test_cases[0].test_id}")
            return True
        else:
            print("✗ No test cases loaded")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loader error: {e}")
        return False


def test_prompt_builder():
    """Test prompt building."""
    print("\nTesting prompt builder...")
    try:
        from src.dataset_loader import load_writing_assistant_dataset
        from src.prompt_builder import WritingAssistantPromptBuilder
        
        # Load one test case
        test_cases = load_writing_assistant_dataset(max_cases=1)
        
        if not test_cases:
            print("✗ No test cases available")
            return False
        
        builder = WritingAssistantPromptBuilder()
        messages = builder.build_messages(test_cases[0])
        
        if messages and len(messages) == 2:
            print("✓ Prompt builder working")
            print(f"  System prompt length: {len(messages[0]['content'])} chars")
            print(f"  User prompt length: {len(messages[1]['content'])} chars")
            return True
        else:
            print("✗ Invalid messages format")
            return False
            
    except Exception as e:
        print(f"✗ Prompt builder error: {e}")
        return False


def test_endpoint():
    """Test endpoint connection."""
    print("\nTesting endpoint connection...")
    try:
        from src.endpoint_client import EndpointClient
        
        client = EndpointClient()
        success = client.test_connection()
        
        if success:
            print("✓ Endpoint connection successful")
            return True
        else:
            print("✗ Endpoint connection failed")
            return False
            
    except Exception as e:
        print(f"✗ Endpoint error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("LLM Evaluation Framework - Validation Script")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Loader", test_dataset_loader),
        ("Prompt Builder", test_prompt_builder),
        ("Endpoint", test_endpoint),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    for i, (name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"{status} - {name}")
    
    total_passed = sum(results)
    total_tests = len(tests)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All validation tests passed! You're ready to run evaluations.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check your configuration and setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
