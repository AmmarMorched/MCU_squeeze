# mcusqueeze/ingestion/validator.py

from typing import Optional
import onnx
from onnx import shape_inference
from pathlib import Path

from mcusqueeze.analysis.graph import extract_ops_from_graph, get_input_output_shapes, get_model_summary
from mcusqueeze.analysis.compatibility import check_target_compatibility  # ✅ Fixed spelling
from mcusqueeze.targets import SUPPORTED_TARGETS, DEFAULT_TARGET, get_available_targets
from mcusqueeze.analysis.memory import (
    get_layer_wise_memory,
    estimate_model_size,
    get_peak_memory_usage,
    get_peak_memory_usage_with_size,
    get_peak_memory_usage_detailed,
)


def validate_onnx(path: str,
                  extract_ops: bool = False,
                  extract_shapes: bool = False,
                  target: str = 'esp32s3',
                  input_height: Optional[int] = None,
                  input_width: Optional[int] = None) -> dict:

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
    except Exception as e:
        warnings.append(f"Shape inference failed: {e}")

    # 6. Dynamic shape check
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_value == 0:
                    warnings.append(f"Dynamic shape in input '{inp.name}' — declare static shape for MCU deployment")

    # 7. Extract operations
    op_analysis = None
    if extract_ops:
        try:
            op_analysis = extract_ops_from_graph(model)
            warnings.append(f"Found {op_analysis['total_ops']} operations across {len(op_analysis['unique_ops'])} types")

            # Check target compatibility for ops
            if target in SUPPORTED_TARGETS:
                target_info = SUPPORTED_TARGETS[target]
                supported_ops = target_info.get('supported_ops', [])
                unsupported_ops = []
                for op_type in op_analysis['unique_ops']:
                    if op_type not in supported_ops:
                        unsupported_ops.append(op_type)
                if unsupported_ops:
                    warnings.append(f"⚠ Unsupported ops on {target}: {', '.join(unsupported_ops)}")

        except Exception as e:
            warnings.append(f"Operation extraction failed: {e}")

    # 8. Extract input/output shapes and memory
    io_shapes = None
    compatibility = None
    model_size = None
    peak_memory_kb = 0
    peak_details = None
    memory = None

    if extract_shapes:
        try:
            # Get input/output shapes
            io_shapes = get_input_output_shapes(model)
            for inp in io_shapes['inputs']:
                warnings.append(f"Input '{inp['name']}': {inp['shape_str']}")
            for out in io_shapes['outputs']:
                warnings.append(f"Output '{out['name']}': {out['shape_str']}")

            # Estimate model size (ONLY ONCE)
            model_size = estimate_model_size(model)
            warnings.append(f"Model size: {model_size['total_size_kb']:.2f} KB ({model_size['total_size_mb']:.2f} MB)")
            warnings.append(f"Parameters: {model_size['total_params']:,}")

            # Estimate memory with appropriate method
            if input_height is not None and input_width is not None:
                warnings.append(f"Using fixed dimensions: {input_height}x{input_width} for memory estimation")
                peak_memory_kb = get_peak_memory_usage_with_size(
                    model,
                    height=input_height,
                    width=input_width
                )
            else:
                # Use batch_size=1 for dynamic shapes
                from mcusqueeze.analysis.dimensions import detect_dynamic_dimensions, get_model_input_shapes
                input_shapes = get_model_input_shapes(model)
                if detect_dynamic_dimensions(input_shapes):
                    warnings.append("⚠️ Model has dynamic dimensions - memory estimated with batch=1 only")
                peak_memory_kb = get_peak_memory_usage(model, batch_size=1)

            if peak_memory_kb > 0:
                warnings.append(f"Peak memory: {peak_memory_kb:.2f} KB ({peak_memory_kb/1024:.2f} MB)")

            # Get detailed peak memory breakdown
            try:
                peak_details = get_peak_memory_usage_detailed(model, batch_size=1)
                largest = peak_details.get('largest_tensor')
                if largest:
                    warnings.append(f"Largest tensor: {largest['op_name']} [{largest['op_type']}] - {largest['size_kb']:.2f} KB")
            except Exception as e:
                warnings.append(f"Detailed peak memory analysis failed: {e}")

            # Get layer-wise memory
            try:
                memory = get_layer_wise_memory(model)
                if memory and memory['total_memory_kb'] > 0:
                    warnings.append(f"Total memory estimate: {memory['total_memory_kb']:.2f} KB ({memory['total_memory_mb']:.2f} MB)")
            except Exception as e:
                warnings.append(f"Layer-wise memory analysis failed: {e}")

            # Check RAM compatibility
            if target in SUPPORTED_TARGETS and peak_memory_kb > 0:
                target_info = SUPPORTED_TARGETS[target]
                ram_kb = target_info.get('ram_kb', 0)
                if ram_kb > 0:
                    ram_usage_percent = (peak_memory_kb / ram_kb) * 100
                    if peak_memory_kb > ram_kb:
                        warnings.append(f"⚠ Model exceeds RAM! Needs {peak_memory_kb:.1f} KB, available {ram_kb} KB")
                    else:
                        warnings.append(f"✅ RAM fits: {peak_memory_kb:.1f} KB / {ram_kb} KB ({ram_usage_percent:.1f}%)")

            # Check flash compatibility
            if target in SUPPORTED_TARGETS and model_size:
                target_info = SUPPORTED_TARGETS[target]
                flash_kb = target_info.get('flash_kb', 0)
                if flash_kb > 0:
                    flash_usage_percent = (model_size['total_size_kb'] / flash_kb) * 100
                    if model_size['total_size_kb'] > flash_kb:
                        warnings.append(f"⚠ Model exceeds flash! Needs {model_size['total_size_kb']:.1f} KB, available {flash_kb} KB")
                    else:
                        warnings.append(f"✅ Flash fits: {model_size['total_size_kb']:.1f} KB / {flash_kb} KB ({flash_usage_percent:.1f}%)")

            # Full compatibility check
            if target in SUPPORTED_TARGETS and op_analysis and model_size:
                compatibility = check_target_compatibility(
                    op_analysis=op_analysis,
                    model_size_kb=model_size['total_size_kb'],
                    memory_kb=peak_memory_kb,
                    target=target
                )

                # Add compatibility warnings/issues to output
                for warning in compatibility.get('warnings', []):
                    warnings.append(f"Compatibility: {warning}")

                if not compatibility.get('compatible', False):
                    for issue in compatibility.get('issues', []):
                        warnings.append(f"Compatibility: {issue}")

        except Exception as e:
            warnings.append(f"Shape extraction failed: {e}")

    if issues:
        return {'valid': False, 'reason': issues[0]}

    return {
        'valid': True,
        'warnings': warnings,
        'opset': opset,
        'model': model,
        'op_analysis': op_analysis,
        'io_shapes': io_shapes,
        'compatibility': compatibility,
        'model_size': model_size,
        'peak_memory_kb': peak_memory_kb,
        'peak_details': peak_details,
        'memory_analysis': memory,
    }