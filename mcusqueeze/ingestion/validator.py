# mcusqueeze/ingestion/validator.py

import onnx
from onnx import shape_inference
from pathlib import Path

from mcusqueeze.analysis.graph import extract_ops_from_graph, get_input_output_shapes,get_model_summary,get_layer_wise_memory


def validate_onnx(path: str, extract_ops:bool = False, extract_shapes:bool = False) -> dict:
    
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
        inferred_model = shape_inference.infer_shapes(model)
        model = inferred_model
        #shape_inference.infer_shapes(model)
    except Exception as e:
        warnings.append(f"Shape inference failed: {e}")
    
    # 6. Dynamic shape check
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_value == 0:
                    warnings.append(f"Dynamic shape in input '{inp.name}' — declare static shape for MCU deployment")
    #7 Extract operations(new - pure extraction, no conversion)
    op_analysis=None
    if extract_ops:
        try:
            op_analysis= extract_ops_from_graph(model)
            #Add summary to warning for vsibility 
            warnings.append(f"Found {op_analysis['total_ops']} operation across {len(op_analysis['unique_ops'])} types")
        except Exception as e:
            warnings.append(f"operation extract failed: {e}")


    #8 Extract input/output shapes
    io_shapes = None
    if extract_shapes:
        try:
            io_shapes = get_input_output_shapes(model)
            for inp in io_shapes['inputs']:
                warnings.append(f"Input '{inp['name']}': {inp['shape_str']}")
            for out in io_shapes['outputs']:
                warnings.append(f"Output '{out['name']}': {out['shape_str']}")

            memory = get_layer_wise_memory(model)
            warnings.append(f"Estimated memory: {memory['total_memory_kb']:.2f} KB ({memory['total_memory_mb']:.2f} MB)")
        except Exception as e :
            warnings.append(f"shape extraction failed: {e}")

    if issues:
        return {'valid': False, 'reason': issues[0]}
    
    return {
        'valid': True,
        'warnings': warnings,
        'opset': opset,
        'model': model,
        'op_analysis': op_analysis,
        'io_shapes': io_shapes,
    }