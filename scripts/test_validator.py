# scripts/test_validator.py

import sys
sys.path.insert(0, '.')

from mcusqueeze.ingestion.validator import validate_onnx

# Test 1 — valid model
print("\n--- Test 1: Valid model ---")
result = validate_onnx("tests/models/valid_model.onnx")
print(result)

# Test 2 — corrupted file
print("\n--- Test 2: Corrupted file ---")
result = validate_onnx("tests/models/corrupted_model.onnx")
print(result)

# Test 3 — old opset
print("\n--- Test 3: Old opset ---")
result = validate_onnx("tests/models/old_opset_model.onnx")
print(result)

# Test 4 — file doesn't exist
print("\n--- Test 4: File not found ---")
result = validate_onnx("tests/models/nonexistent.onnx")
print(result)