# mcusqueeze/ingestion/validator.py

import onnx
from onnx import shape_inference
from pathlib import Path

def validate_onnx(path: str) -> dict:
    
    issues = []
    warnings = []
    
    # 1. File exists
    if not Path(path).exists():
        return {'valid': False, 'reason': f"File not found: {path}"}
    
    # 2. Load + basic structure check
    try:
        model = onnx.load(path)
    except Exception as e:
        return {'valid': False, 'reason': f"Cannot load file: {e}"}
    
    # 3. Graph validity
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        return {'valid': False, 'reason': f"Invalid graph: {e}"}
    
    # 4. Opset check
    opset = model.opset_import[0].version
    if opset < 11:
        issues.append(f"Opset {opset} too old — minimum 11 required")
    
    # 5. Shape inference
    try:
        shape_inference.infer_shapes(model)
    except Exception as e:
        warnings.append(f"Shape inference failed: {e}")
    
    # 6. Dynamic shape check
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_value == 0:
                    warnings.append(f"Dynamic shape in input '{inp.name}' — declare static shape for MCU deployment")
    
    if issues:
        return {'valid': False, 'reason': issues[0]}
    
    return {
        'valid': True,
        'warnings': warnings,
        'opset': opset,
        'model': model
    }