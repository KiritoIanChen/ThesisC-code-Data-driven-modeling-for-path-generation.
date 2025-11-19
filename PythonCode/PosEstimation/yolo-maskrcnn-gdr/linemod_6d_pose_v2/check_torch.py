import sys
try:
    import torch
    print("✓ PyTorch installed successfully!")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
except Exception as e:
    print(f"✗ PyTorch not available: {e}")
    sys.exit(1)

