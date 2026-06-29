# scripts/generate_test_models.py

import torch
import torch.nn as nn
import onnx
import tensorflow as tf

# 1. Simple valid model
class TinyModel(nn.Module):
    def forward(self, x):
        return x * 2

keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

keras_model.save("tests/models/valid_model.h5")
print("✓ valid_model.h5 created")

model = TinyModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "tests/models/valid_model.onnx",
    opset_version=17
)
print("✓ valid_model.onnx created")


# 2. Simulate a corrupted file
with open("tests/models/corrupted_model.onnx", "w") as f:
    f.write("this is not an onnx file")
print("✓ corrupted_model.onnx created")


# 3. Old opset model
torch.onnx.export(
    model,
    dummy_input,
    "tests/models/old_opset_model.onnx",
    opset_version=9       # too old
)
print("✓ old_opset_model.onnx created")