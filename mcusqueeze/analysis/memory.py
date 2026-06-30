# mcusqueeze/analysis/memory.py

import numpy as np
import onnx
from typing import Dict
from collections import Counter

from .graph import extract_ops_from_graph, get_tensor_shapes


def get_layer_wise_memory(model: onnx.ModelProto) -> Dict:
    """
    Calculate memory usage per layer based on tensor shapes.
    """

    analysis = extract_ops_from_graph(model)
    tensor_shapes = analysis['tensor_shapes']

    memory_by_layer = []
    total_memory = 0

    for op_detail in analysis['op_details']:
        layer_memory = 0
        
        # Sum up memory for input tensors
        for shape in op_detail['input_shapes']:
            if shape:
                bytes_per_param = 4  # float32
                # Only count static dimensions (positive values)
                static_dims = [d for d in shape if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    layer_memory += n_params * bytes_per_param

        # Sum up memory for output tensors
        for shape in op_detail['output_shapes']:
            if shape:
                bytes_per_param = 4  # float32
                # Only count static dimensions (positive values)
                static_dims = [d for d in shape if d > 0]
                if static_dims:
                    n_params = np.prod(static_dims)
                    layer_memory += n_params * bytes_per_param

        total_memory += layer_memory

        memory_by_layer.append({
            'name': op_detail['name'],
            'type': op_detail['type'],
            'memory_bytes': layer_memory,
            'memory_kb': layer_memory / 1024,
        })

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


def get_memory_summary(model: onnx.ModelProto) -> str:
    """
    Get a human-readable summary of model memory usage.
    """
    
    memory = get_layer_wise_memory(model)
    size = estimate_model_size(model)
    
    lines = []
    lines.append("📦 Memory & Size Summary:")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Model Size:")
    lines.append(f"  Total parameters: {size['total_params']:,}")
    lines.append(f"  Parameter size:   {size['total_params_kb']:.2f} KB ({size['total_params_mb']:.2f} MB)")
    lines.append(f"  Graph overhead:   {size['graph_size_kb']:.2f} KB")
    lines.append(f"  Total model size: {size['total_size_kb']:.2f} KB ({size['total_size_mb']:.2f} MB)")
    lines.append("")
    lines.append("Activation Memory:")
    lines.append(f"  Total memory:     {memory['total_memory_kb']:.2f} KB ({memory['total_memory_mb']:.2f} MB)")
    lines.append("")
    
    # Show largest memory-consuming layers
    if memory['memory_by_layer']:
        lines.append("Largest layers (by memory):")
        sorted_layers = sorted(memory['memory_by_layer'], 
                              key=lambda x: x['memory_kb'], reverse=True)
        top_n = min(5, len(sorted_layers))
        for i in range(top_n):
            layer = sorted_layers[i]
            lines.append(f"  {layer['name']} ({layer['type']}): {layer['memory_kb']:.2f} KB")
    
    return "\n".join(lines)