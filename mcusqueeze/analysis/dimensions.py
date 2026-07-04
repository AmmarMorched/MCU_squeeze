import onnx
from typing import Dict, List, Tuple, Optional


def get_model_input_shapes(model: onnx.ModelProto) -> List[Dict]:
    """
    Extract input shapes from the model.
    """
    graph = model.graph
    input_shapes = []
    
    for input_tensor in graph.input:
        shape = []
        has_dynamic = False
        
        if input_tensor.type.tensor_type.HasField('shape'):
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(int(dim.dim_value))
                else:
                    shape.append(-1)  # Dynamic dimension
                    has_dynamic = True
        
        input_shapes.append({
            'name': input_tensor.name,
            'shape': shape,
            'has_dynamic': has_dynamic,
            'shape_str': format_shape_for_display(shape)
        })
    
    return input_shapes


def detect_dynamic_dimensions(input_shapes: List[Dict]) -> bool:
    """
    Check if any input has dynamic dimensions.
    """
    for shape_info in input_shapes:
        if shape_info['has_dynamic']:
            return True
    return False


def suggest_default_dimensions(model: onnx.ModelProto, target: str) -> Tuple[int, int]:
    """
    Suggest intelligent default dimensions based on model analysis.
    
    Returns:
        Tuple of (height, width)
    """
    # Try to extract from model metadata
    graph = model.graph
    
    # Check if there are any fixed dimensions we can use
    for input_tensor in graph.input:
        if input_tensor.type.tensor_type.HasField('shape'):
            dims = input_tensor.type.tensor_type.shape.dim
            # If we have at least one fixed dimension, use it as hint
            fixed_dims = []
            for dim in dims:
                if dim.HasField('dim_value'):
                    fixed_dims.append(int(dim.dim_value))
            
            # For image models, we typically have [batch, channels, height, width]
            # or [batch, height, width, channels]
            if len(fixed_dims) >= 2:
                # Try to guess height and width
                # Common patterns: [batch, channels, height, width] or [batch, height, width, channels]
                if len(dims) == 4:
                    # Check if it's channels-first or channels-last
                    # This is a heuristic
                    if fixed_dims[1] in [1, 3]:  # Likely channels-first (NCHW)
                        if len(fixed_dims) >= 4:
                            return fixed_dims[2], fixed_dims[3]
                    elif fixed_dims[-1] in [1, 3]:  # Likely channels-last (NHWC)
                        if len(fixed_dims) >= 3:
                            return fixed_dims[1], fixed_dims[2]
    
    # Default based on target
    # For YOLO-like models, 640 is common
    # For mobile models, 224 or 320
    
    # Check if it's likely a YOLO model (detect by ops)
    has_yolo_ops = False
    for node in graph.node:
        if node.op_type in ['Resize', 'Concat']:
            # YOLO uses many concat and resize ops
            has_yolo_ops = True
    
    if has_yolo_ops:
        # YOLO default
        return 640, 640
    else:
        # General vision model default
        return 224, 224


def format_shape_for_display(shape: List[int]) -> str:
    """
    Format shape for display, replacing -1 with ?.
    """
    if not shape:
        return "()"
    parts = []
    for d in shape:
        if d == -1:
            parts.append("?")
        else:
            parts.append(str(d))
    return f"({','.join(parts)})"