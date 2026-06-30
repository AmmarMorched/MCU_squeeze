import torch
import ultralytics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version used by PyTorch: {torch.version.cuda}")
print(f"Ultralytics version: {ultralytics.__version__}")