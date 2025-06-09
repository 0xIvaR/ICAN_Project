"""
Simple test script to verify ICAN installation and basic functionality
"""

def test_imports():
    """Test basic imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found - install with: pip install torch")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    return True

def test_config():
    """Test configuration module"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        print(f"✓ Model: {Config.MODEL_NAME}")
        print(f"✓ Device: {Config.DEVICE}")
        print(f"✓ Image size: {Config.IMG_SIZE}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_simple_model():
    """Test basic model creation without external dependencies"""
    print("\nTesting simple model...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.fc = nn.Linear(16, 2)
                
            def forward(self, x):
                x = self.conv(x)
                x = nn.functional.adaptive_avg_pool2d(x, 1)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = SimpleModel()
        test_input = torch.randn(1, 3, 32, 32)
        output = model(test_input)
        print(f"✓ Simple model test passed")
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("ICAN Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_simple_model
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All tests passed! ICAN is ready to use.")
        print("\nNext steps:")
        print("1. Install remaining dependencies: pip install -r requirements.txt")
        print("2. Run demo: python demo.py")
        print("3. Start training: python main.py")
    else:
        print("❌ Some tests failed. Please install missing dependencies.")

if __name__ == "__main__":
    main() 