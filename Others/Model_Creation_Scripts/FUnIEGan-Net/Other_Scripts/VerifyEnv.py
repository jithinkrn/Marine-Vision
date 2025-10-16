import torch
import sys

# Check Python version
python_version = sys.version
print(f"Python version: {python_version}")

# Check PyTorch version
pytorch_version = torch.__version__
print(f"PyTorch version: {pytorch_version}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# If CUDA is available, print CUDA device details
if cuda_available:
    cuda_device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device: {cuda_device_name}")
else:
    print("CUDA device: Not available")

