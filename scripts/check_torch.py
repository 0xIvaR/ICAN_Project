try:
    import torch
    print('✅ PyTorch installed successfully!')
    print(f'Version: {torch.__version__}')
    print(f'Device: {torch.device("cpu")}')
except Exception as e:
    print(f'❌ Error: {e}')
