import onnxruntime as ort
import numpy as np

# Load quantized model
session = ort.InferenceSession('quantized/quantized_model.onnx')

# Run inference
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {'images': input_data})