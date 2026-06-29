# mcusqueeze/ingestion/h5_converter.py

import tf2onnx
import tensorflow as tf
import onnx
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def convert_h5_to_onnx(h5_path: str, output_path: str = None) -> str:
    """
    Convert a Keras .h5 model to .onnx
    Returns path to the converted .onnx file
    """
    
    h5_path = Path(h5_path)
    
    # Default output path — same folder, same name, different extension
    if output_path is None:
        output_path = h5_path.with_suffix('.onnx')
    
    output_path = Path(output_path)
    
    # 1. Load the Keras model
    try:
        model = tf.keras.models.load_model(str(h5_path), compile=False)
    except Exception as e:
        raise ValueError(f"Cannot load .h5 model: {e}")
    
    # 2. Build input signature from model input shape directly
    try:
        input_signature = [
            tf.TensorSpec(
                shape=[dim if dim is not None else None for dim in inp.shape],
                dtype = inp.dtype,
                name=f"input_{i}"
            )
            for i, inp in enumerate(model.inputs)
        ]
    except Exception as e:
        raise ValueError(f"Cannot read input signature:{e}")
    
    # 3. Convert
    try:
        @tf.function(input_signature=input_signature)
        def model_fn(*args):
            return model(*args)
          
        model_proto, _ = tf2onnx.convert.from_function(
            model_fn,
            input_signature=input_signature,
            opset=17,                    # target opset
            output_path=str(output_path)
        )
    except Exception as e:
        raise ValueError(f"Conversion failed: {e}")
    
    return str(output_path)