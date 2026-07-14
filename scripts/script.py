import torch
import torchvision

# Load a pre-trained model (e.g., ResNet-18)
model = torchvision.models.resnet18(pretrained=True).eval()
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx")