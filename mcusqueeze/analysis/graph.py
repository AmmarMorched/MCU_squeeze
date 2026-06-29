import numpy as np
import onnx
from collections import Counter
from typing import Dict, Set

def extract_ops_from_graph(model: onnx.ModelProto) -> Dict:
    """
    Extract all operations from ONNX model graph.

    Returns: 
        Dictionary with:
            - op_counts: {operation_type: count}
            - unique_ops: set of unique operation types
            - tottal_ops: total number of operations
            - op_list: list of all operation types in order
            - op_details: detailed info including shapes
    """

    graph = model.graph
    #Count operation
    op_counts = Counter()
    op_list = []
    op_details= []

    tensor_shapes = get_tensor_shapes(model)

    #Traverse every node in the graph
    for node in graph.node:
        op_type = node.op_type
        op_counts[op_type]+=1
        op_list.append(op_type)

        #get input shapes
        input_shapes = []
        for input_name in node.input:
            if input_name in tensor_shapes:
                input_shapes.append(tensor_shapes[input_name])
            else:  input_shapes.append(None)

        #get output shapes 
        output_shapes= []
        for output_name in node.output:
            if output_name in tensor_shapes:
                output_shapes.append(tensor_shapes[output_name])
            else: output_shapes.append(None)

        # store details info 
        op_details.append({
            'name': node.name or f"{op_type}_{len(op_details)}",
            'type': op_type,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
        })
            

    return{
        'op_counts': dict(op_counts),
        'unique_ops':set(op_counts.keys()),
        'total_ops': len(op_list),
        'op_list':op_list,
        'op_details':op_details,
        'tensor_shapes': tensor_shapes,
    }

def get_shape_from_value_info(value_info)->list[int]:
    shape=[]
    if value_info.type.tensor_type.HasField('shape'):
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(int(dim.dim_value))
            else:
                shape.append(-1)
    return shape if shape else None

def get_tensor_shapes(model: onnx.ModelProto)->Dict[str, list]:
    """
    Extract all tensor shapes from the model.
    
    Returns a dictionary mapping tensor names to their shapes.
    """

    graph= model.graph
    tensor_shapes={}

    #get shape from value info (inputs, outputs, intermediate tensors)
    for value_info in graph.value_info:
        tensor_name = value_info.name
        shape = get_shape_from_value_info(value_info)
        if shape: 
            tensor_shapes[tensor_name] = shape

    #get shapes from inputs
    for input_tensor in graph.input:
        tensor_name = input_tensor.name
        shape = get_shape_from_value_info(input_tensor)
        if shape:
            tensor_shapes[tensor_name] = shape

    #get shape from outputs
    for output_tensor in graph.output:
        tensor_name = output_tensor.name
        shape = get_shape_from_value_info(output_tensor)
        if (shape):
            tensor_shapes[tensor_name] = shape


    #get shape from initializers (weight, biases)
    for init in graph.initializer:
        tensor_name= init.name
        shape = list(init.dims)
        tensor_shapes[tensor_name] = shape
    

    return tensor_shapes

def format_shape(shape)->str:
    if shape is None:
        return "?"
    return f"({','.join(str(d) if d != -1 else '?' for d in shape)})"

def get_model_summary(model: onnx.ModelProto) -> str:
    """
    Get human readable summary of operation.
    print what ops exist and how many.
    """
    
    analysis = extract_ops_from_graph(model)

    lines = []
    lines.append("📊 Operations in model: ")
    lines.append("="* 50)
    lines.append(f"Total operations: {analysis['total_ops']}")
    lines.append(f"Unique op types: {len(analysis['unique_ops'])}")
    lines.append("")
    lines.append("Operation breakdown")
    lines.append("="*50)

    for op_detail in analysis['op_details']:
        op_type = op_detail['type']
        op_name = op_detail['name']
        input_shapes = op_detail['input_shapes']
        output_shapes = op_detail['output_shapes']

        # Format shapes for display
        input_shape_str = " → ".join([format_shape(s) for s in input_shapes])
        output_shape_str = " → ".join([format_shape(s) for s in output_shapes])

        lines.append(f"  [{op_name}] {op_type}")
        lines.append(f"    Inputs:  {input_shape_str}")
        lines.append(f"    Outputs: {output_shape_str}")

    lines.append("")
    lines.append("Operation counts: ")
    lines.append("-"*50)

    #sort by count 
    for op_type, count in sorted (analysis['op_counts'].items(), key = lambda x: x[1], reverse=True):
        lines.append(f" {op_type}: {count}")

    return "\n".join(lines)


def get_input_output_shapes(model: onnx.ModelProto) -> Dict:
    """
    Extract just the input and output shapes of the entire model.
    
    Returns:
        Dictionary with 'inputs' and 'outputs' shapes
    """

    graph = model.graph

    inputs = []
    for input_tensor in graph.input:
        shape = get_shape_from_value_info(input_tensor)
        inputs.append({
            'name': input_tensor.name,
            'shape': shape,
            'shape_str': format_shape(shape)
        })

    outputs=[]
    for output_tensor in graph.output:
        shape = get_shape_from_value_info(output_tensor)
        outputs.append({
            'name':output_tensor.name,
            'shape':shape,
            'shape_str': format_shape(shape)
        })

    return{
        'inputs': inputs,
        'outputs': outputs,
        }

def get_layer_wise_memory(model :onnx.ModelProto) ->Dict:
    """
    Calculate memory usage per layer based on tensor shapes.
    """

    analysis = extract_ops_from_graph(model)
    tensor_shapes= analysis['tensor_shapes']

    memory_by_layer = []
    total_memory = 0
    static_dims = [d for d in shape if d > 0]

    for op_detail in analysis['op_details']:
        layer_memory = 0
        #sum up memory for input tensors
        for shape in op_detail['input_shapes']:
            if shape:
                bytes_per_param = 4
                n_params = np.prod(static_dims)
                layer_memory += n_params * bytes_per_param

        #sum up memory for output
        for shape in op_detail['output_shapes']:
            if shape:
                bytes_per_param = 4
                n_params = np.prod(static_dims)
                layer_memory += n_params*bytes_per_param


        total_memory += layer_memory

        memory_by_layer.append({
            'name': op_detail['name'],
            'type': op_detail['type'],
            'memory_bytes': layer_memory,
            'memory_kb': layer_memory / 1024,
        })

    return{
        'memory_by_layer': memory_by_layer,
        'total_memory_bytes': total_memory,
        'total_memory_kb': total_memory / 1024,
        'total_memory_mb': total_memory / (1024 * 1024),
        }


def get_input_output_summary(model: onnx.ModelProto)-> str:
    """
    Get a summary of model inputs and outputs only.
    """

    shapes = get_input_output_shapes(model)

    lines = []
    lines.append("📋 Model Inputs/Outputs:")
    lines.append("=" * 40)
    
    lines.append("Inputs:")
    for inp in shapes['inputs']:
        lines.append(f"  {inp['name']}: {inp['shape_str']}")
    
    lines.append("")
    lines.append("Outputs:")
    for out in shapes['outputs']:
        lines.append(f"  {out['name']}: {out['shape_str']}")
    
    return "\n".join(lines)