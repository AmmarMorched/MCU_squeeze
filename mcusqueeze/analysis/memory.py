# mcusqueeze/analysis/memory.py

import numpy as np
import onnx
from typing import Dict, Optional
from collections import Counter

from .graph import extract_ops_from_graph, format_shape, get_tensor_shapes


def get_layer_wise_memory(model: onnx.ModelProto) -> Dict:
    """
    Calculate memory usage per layer based on tensor shapes.
    """

    analysis = extract_ops_from_graph(model)
    #tensor_shapes = analysis['tensor_shapes']

    memory_by_layer = []
    total_memory = 0

    for op_detail in analysis['op_details']:
        layer_memory = 0
        
        # Memory is primarily determined by output tensors

        # Sum up memory for output tensors
        for shape in op_detail['output_shapes']:
            if shape:
                bytes_per_param = 4  # float32
                # Only count static dimensions (positive values)
                static_dims = [d for d in shape if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    layer_memory += n_params * bytes_per_param


        # Sum up memory for input tensors
        for shape in op_detail['input_shapes']:
            if shape:
                bytes_per_param = 4  # float32
                # Only count static dimensions (positive values)
                static_dims = [d for d in shape if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    if n_params > 1000:
        
                        layer_memory += n_params * bytes_per_param

        

        total_memory += layer_memory

        memory_by_layer.append({
            'name': op_detail['name'],
            'type': op_detail['type'],
            'memory_bytes': layer_memory,
            'memory_kb': layer_memory / 1024,
        })
    # Sort by memory usage (largest first)
    memory_by_layer.sort(key=lambda x: x['memory_kb'], reverse=True)

    return {
        'memory_by_layer': memory_by_layer,
        'total_memory_bytes': total_memory,
        'total_memory_kb': total_memory / 1024,
        'total_memory_mb': total_memory / (1024 * 1024),
    }


def estimate_model_size(model: onnx.ModelProto) -> Dict:
    """
    Estimate the total size of the model in KB.
    """
    
    graph = model.graph
    
    total_params = 0
    param_details = []
    
    # Calculate parameter sizes
    for init in graph.initializer:
        n_params = np.prod(init.dims)
        total_params += n_params
        
        size_bytes = n_params * 4  # float32
        size_kb = size_bytes / 1024
        
        param_details.append({
            'name': init.name,
            'shape': list(init.dims),
            'n_params': int(n_params),
            'size_kb': size_kb,
        })

    #sort by size (largest first )
    param_details.sort(key=lambda x: x['size_kb'], reverse=True)
    
    # Graph structure overhead (rough estimate)
    graph_size_kb = len(graph.node) * 0.5  # ~0.5KB per node
    
    total_params_kb = sum(p['size_kb'] for p in param_details)
    total_size_kb = total_params_kb + graph_size_kb
    
    return {
        'total_params': total_params,
        'total_params_kb': total_params_kb,
        'total_params_mb': total_params_kb / 1024,
        'graph_size_kb': graph_size_kb,
        'total_size_kb': total_size_kb,
        'total_size_mb': total_size_kb / 1024,
        'param_details': param_details,
        'n_parameters': len(param_details),
        'n_nodes': len(graph.node),
    }

def get_peak_memory_usage(model: onnx.ModelProto, batch_size: int = 1)-> float:
    """
    Calculate the peak memory usage (maximum memory at any point during inference).
    This is more accurate than summing all tensors.
    """
    analysis = extract_ops_from_graph(model)
    max_tensor_size = 0
    max_tensor_shape = None

    for op_detail in analysis['op_details']:
        for shape in op_detail['output_shapes']:
            if shape:
                dims = [batch_size if d ==-1 else d for d in shape]
                static_dims = [d for d in dims if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    size_bytes = n_params*4
                    if size_bytes > max_tensor_size:
                        max_tensor_size = size_bytes
                        max_tensor_shape = shape

    #check inputs
    for op_detail in analysis['op_details']:
        for shape in op_detail['input_shapes']:
            if shape:
                dims = [batch_size if d == -1 else d for d in shape]
                static_dims = [d for d in dims if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    size_bytes = n_params*4
                    if size_bytes > max_tensor_size:
                        max_tensor_size = size_bytes
                        max_tensor_shape = shape

    #peak memory in approximately the largest tensor plus some overhead
    peak_memory_kb = (max_tensor_size) / 1024  

    return peak_memory_kb


def get_memory_summary(model: onnx.ModelProto, batch_size: int = 1) -> str:
    """
    Get a human-readable summary of model memory usage.
    """
    
    memory = get_layer_wise_memory(model)
    size = estimate_model_size(model)
    peak_details = get_peak_memory_usage_detailed(model, batch_size)

    lines = []
    lines.append("📦 Memory & Size Summary:")
    lines.append("=" * 40)
    lines.append("")
    lines.append(" Model Size (Flash):")
    lines.append(f"  Total parameters: {size['total_params']:,}")
    lines.append(f"  Parameter size:   {size['total_params_kb']:.2f} KB ({size['total_params_mb']:.2f} MB)")
    lines.append(f"  Graph overhead:   {size['graph_size_kb']:.2f} KB")
    lines.append(f"  Total model size: {size['total_size_kb']:.2f} KB ({size['total_size_mb']:.2f} MB)")
    lines.append("")
    lines.append("Activation Memory (RAM):")
    lines.append(f"  Peak memory:       {peak_details['peak_kb']:.2f} KB ({peak_details['peak_mb']:.2f} MB)")
    lines.append(f"  Total tensors:     {peak_details['total_tensors']}")
    lines.append("")

     # Largest tensor
    largest = peak_details.get('largest_tensor')
    if largest:
        lines.append("Largest Tensor:")
        lines.append(f"  Operation: {largest['op_name']} [{largest['op_type']}]")
        lines.append(f"  Shape:     {format_shape(largest['shape'])}")
        lines.append(f"  Size:      {largest['size_kb']:.2f} KB ({largest['size_mb']:.2f} MB)")
        lines.append("")
    
    top_tensors = peak_details.get('top_tensors', [])
    if top_tensors:
        lines.append("Top 5 Largest Tensors:")
        for i, tensor in enumerate(top_tensors[:5], 1):
            lines.append(f"  {i}. {tensor['op_name']} [{tensor['op_type']}]")
            lines.append(f"     Shape: {format_shape(tensor['shape'])}")
            lines.append(f"     Size:  {tensor['size_kb']:.2f} KB")


    return "\n".join(lines)

# mcusqueeze/analysis/memory.py

def get_peak_memory_usage_with_size(
    model: onnx.ModelProto, 
    height: Optional[int] = None, 
    width: Optional[int] = None
) -> float:
    """
    Calculate peak memory with specific input dimensions.
    
    If height/width not provided, uses existing fixed dimensions.
    """
    analysis = extract_ops_from_graph(model)
    max_tensor_size = 0
    
    def get_effective_shape(shape):
        """Replace dynamic dimensions with provided values."""
        if not shape:
            return None
        result = []
        for d in shape:
            if d <= 0:  # Dynamic dimension
                # Try to infer if it's height or width
                # This is a heuristic - for NCHW [batch, channels, height, width]
                # or NHWC [batch, height, width, channels]
                if len(result) == 2 and height is not None:  # Height dimension
                    result.append(height)
                elif len(result) == 3 and width is not None:  # Width dimension
                    result.append(width)
                else:
                    # Unknown dynamic dimension, use 1 as fallback
                    result.append(1)
            else:
                result.append(d)
        return result
    
    # Check all tensors
    for op_detail in analysis['op_details']:
        for shape in op_detail['output_shapes']:
            if shape:
                effective_shape = get_effective_shape(shape)
                if effective_shape:
                    n_params = np.prod(effective_shape)
                    size_bytes = n_params * 4
                    if size_bytes > max_tensor_size:
                        max_tensor_size = size_bytes
    
    return max_tensor_size / 1024



def get_peak_memory_usage_detailed(model: onnx.ModelProto, batch_size: int = 1) -> Dict:
    """
    Calculate detailed peak memory usage with tensor information.
    
    Returns:
        Dictionary with:
            - peak_kb: Peak memory in KB
            - peak_mb: Peak memory in MB
            - largest_tensor: Info about largest tensor
            - top_tensors: Top 5 largest tensors
    """
    analysis = extract_ops_from_graph(model)
    
    tensor_sizes = []
    max_tensor_size = 0
    max_tensor_info = None
    
    # Check output tensors
    for op_detail in analysis['op_details']:
        for idx, shape in enumerate(op_detail['output_shapes']):
            if shape:
                dims = [batch_size if d == -1 else d for d in shape]
                static_dims = [d for d in dims if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    size_bytes = n_params * 4  # float32
                    size_kb = size_bytes / 1024
                    
                    tensor_info = {
                        'op_name': op_detail['name'],
                        'op_type': op_detail['type'],
                        'tensor_type': 'output',
                        'tensor_index': idx,
                        'shape': shape,
                        'effective_shape': dims,
                        'size_kb': size_kb,
                        'size_mb': size_kb / 1024,
                    }
                    tensor_sizes.append(tensor_info)
                    
                    if size_bytes > max_tensor_size:
                        max_tensor_size = size_bytes
                        max_tensor_info = tensor_info
    
    # Check input tensors (weights, biases)
    for op_detail in analysis['op_details']:
        for idx, shape in enumerate(op_detail['input_shapes']):
            if shape:
                dims = [batch_size if d == -1 else d for d in shape]
                static_dims = [d for d in dims if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    size_bytes = n_params * 4  # float32
                    size_kb = size_bytes / 1024
                    
                    tensor_info = {
                        'op_name': op_detail['name'],
                        'op_type': op_detail['type'],
                        'tensor_type': 'input',
                        'tensor_index': idx,
                        'shape': shape,
                        'effective_shape': dims,
                        'size_kb': size_kb,
                        'size_mb': size_kb / 1024,
                    }
                    tensor_sizes.append(tensor_info)
                    
                    if size_bytes > max_tensor_size:
                        max_tensor_size = size_bytes
                        max_tensor_info = tensor_info
    
    # Sort by size (largest first)
    tensor_sizes.sort(key=lambda x: x['size_kb'], reverse=True)
    
    return {
        'peak_kb': max_tensor_size / 1024,
        'peak_mb': max_tensor_size / (1024 * 1024),
        'largest_tensor': max_tensor_info,
        'top_tensors': tensor_sizes[:10],  # Top 10 largest tensors
        'total_tensors': len(tensor_sizes),
    }


def get_activation_memory_breakdown(model: onnx.ModelProto, batch_size: int = 1) -> Dict:
    """
    Get detailed breakdown of activation memory per layer.
    """
    analysis = extract_ops_from_graph(model)
    
    activation_memory = []
    total_activation_memory = 0
    
    for op_detail in analysis['op_details']:
        layer_activation = 0
        
        # Activation memory is primarily output tensors
        for shape in op_detail['output_shapes']:
            if shape:
                dims = [batch_size if d == -1 else d for d in shape]
                static_dims = [d for d in dims if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    size_kb = (n_params * 4) / 1024
                    layer_activation += size_kb
        
        total_activation_memory += layer_activation
        
        if layer_activation > 0:
            activation_memory.append({
                'op_name': op_detail['name'],
                'op_type': op_detail['type'],
                'activation_kb': layer_activation,
                'activation_mb': layer_activation / 1024,
            })
    
    # Sort by memory usage
    activation_memory.sort(key=lambda x: x['activation_kb'], reverse=True)
    
    return {
        'total_activation_kb': total_activation_memory,
        'total_activation_mb': total_activation_memory / 1024,
        'layers': activation_memory,
        'top_layers': activation_memory[:10],
    }