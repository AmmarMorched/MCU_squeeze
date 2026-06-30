# mcusqueeze/ingestion/validator.py

import onnx
from onnx import shape_inference
from pathlib import Path

from mcusqueeze.analysis.graph import extract_ops_from_graph, get_input_output_shapes,get_model_summary
from mcusqueeze.analysis.compatibility import check_target_compatibility
from mcusqueeze.targets import SUPPORTED_TARGETS,DEFAULT_TARGET,get_available_targets
from mcusqueeze.analysis.memory import get_layer_wise_memory, estimate_model_size  


def validate_onnx(path: str, extract_ops:bool = False, extract_shapes:bool = False, target:str='esp32s3') -> dict:
    
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
            # === Check target compatibility for ops ===
            if target in SUPPORTED_TARGETS:
                target_info = SUPPORTED_TARGETS[target]
                supported_ops = target_info.get('supported_ops',[])
                unsupported_ops=[]
                for op_type in op_analysis['unique_ops']:
                    if op_type not in supported_ops:
                        unsupported_ops.append(op_type)
                if unsupported_ops:
                    warnings.append(f"⚠ Unsupported ops on {target}: {', '.join(unsupported_ops)}")
        
        except Exception as e:
            warnings.append(f"operation extract failed: {e}")


    #8 Extract input/output shapes
    io_shapes = None
    compatibility = None
    if extract_shapes:
        try:
            io_shapes = get_input_output_shapes(model)
            for inp in io_shapes['inputs']:
                warnings.append(f"Input '{inp['name']}': {inp['shape_str']}")
            for out in io_shapes['outputs']:
                warnings.append(f"Output '{out['name']}': {out['shape_str']}")

            # Estimate model size
            model_size = estimate_model_size(model)
            warnings.append(f"Model size: {model_size['total_size_kb']:.2f} KB ({model_size['total_size_mb']:.2f} MB)")
            warnings.append(f"Parameters: {model_size['total_params']:,}")

            # Estimate memory
            memory = get_layer_wise_memory(model)
            if memory['total_memory_kb'] >0:
                warnings.append(f"Estimated activation memory: {memory['total_memory_kb']:.2f} KB")

                # === Check RAM compatibility ===
            if target in SUPPORTED_TARGETS and op_analysis:
                compatibility = check_target_compatibility(
                    op_analysis=op_analysis,
                    model_size_kb=model_size['total_size_kb'],
                    memory_kb=memory['total_memory_kb'],
                    target=target
                    )
            
            # Add compatibility warnings/issues to output
                for warning in compatibility.get('warnings', []):
                    warnings.append(f"Compatibility: {warning}")
            
                if not compatibility.get('compatible', False):
                    for issue in compatibility.get('issues', []):
                        warnings.append(f"Compatibility: {issue}")
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
        'comptability':compatibility,
        
    }