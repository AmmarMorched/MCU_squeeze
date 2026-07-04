# mcusqueeze/ingestion/pt_converter.py

import torch
import onnx
from pathlib import Path
import numpy as np
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def is_yolo_model(pt_path: str) -> bool:
    """
    Detect if a .pt file is a YOLO model by checking its contents.
    
    Returns:
        True if it's a YOLO model, False otherwise
    """
    try:
        # Try to load with ultralytics first (most reliable)
        try:
            from ultralytics import YOLO
            YOLO(str(pt_path))
            return True
        except ImportError:
            pass
        except Exception:
            pass
        
        # Alternative: Check file contents
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            
            if isinstance(data, dict):
                yolo_keys = ['model', 'names', 'nc', 'stride', 'pt']
                if any(key in data for key in yolo_keys):
                    return True
                if 'model' in data and hasattr(data['model'], 'model'):
                    return True
                    
            if hasattr(data, 'model') and hasattr(data, 'names'):
                return True
                
        except Exception:
            pass
            
        return False
        
    except Exception:
        return False


def convert_pt_to_onnx(pt_path: str, output_path: str = None, input_shape: tuple = (1, 3, 224, 224)) -> str:
    """
    Convert a PyTorch .pt model to .onnx.
    
    Args:
        pt_path: Path to .pt file
        output_path: Output path for ONNX file (optional)
        input_shape: Input shape for the model (batch, channels, height, width)
    
    Returns:
        Path to the converted .onnx file
    """
    
    pt_path = Path(pt_path)
    
    # ✅ FIXED: Handle output_path correctly
    if output_path is None:
        # Default: same folder, .onnx extension
        output_path = pt_path.with_suffix('.onnx')
    else:
        output_path = Path(output_path)

    # === STEP 1: AUTO-DETECT YOLO ===
    if is_yolo_model(pt_path):
        print("🟢 YOLO model detected - using Ultralytics export")
        
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model from {pt_path}...")
            model = YOLO(str(pt_path))
            
            print(f"Exporting to ONNX at {output_path}...")
            model.export(
                format='onnx',
                imgsz=640,
                opset=17,
                simplify=True,
                dynamic=True,
                device='cpu'
            )
            
            # Handle the default filename
            default_path = pt_path.parent / f"{pt_path.stem}.onnx"
            if default_path != output_path:
                shutil.move(str(default_path), str(output_path))
            
            print(f"✅ Conversion complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            raise ImportError(
                "Ultralytics is required for YOLO models.\n"
                "Install it with: pip install ultralytics"
            )
        except Exception as e:
            raise ValueError(f"YOLO conversion failed: {e}")
    
    # === STEP 2: STANDARD PYTORCH MODEL ===
    print("🟣 Standard PyTorch model detected - using torch.onnx.export")
    
    # Load the model
    try:
        # Try loading with weights_only=True first
        model = torch.load(pt_path, map_location='cpu', weights_only=True)
        print("✅ Model loaded with weights_only=True")
    except Exception:
        # Fall back to weights_only=False
        print("⚠️ Warning: Falling back to weights_only=False. This is only safe for files from trusted sources.")
        try:
            model = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Cannot load .pt model: {e}")
    
    # Check if it's a state_dict (weights only)
    if isinstance(model, dict):
        raise ValueError(
            f"'{pt_path}' contains only model weights (state_dict).\n"
            f"Please provide a complete model object or use a model class with load_state_dict().\n"
            f"Tips:\n"
            f"  - For YOLO models, install ultralytics: pip install ultralytics\n"
            f"  - For other models, load with the model class first"
        )
    
    # Set to eval mode
    model.eval()
    
    # Create dummy input for tracing
    try:
        dummy_input = torch.randn(*input_shape)
        print(f"📐 Using input shape: {input_shape}")
    except Exception as e:
        raise ValueError(f"Cannot create dummy input: {e}")
    
    # ✅ Export to ONNX (ONLY ONCE!)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Conversion complete: {output_path}")
        return str(output_path)
        
    except Exception as e:
        raise ValueError(f"Conversion to ONNX failed: {e}")


def load_pt_model_with_class(pt_path: str, model_class, input_shape: tuple = (1, 3, 224, 224)) -> str:
    """
    Load a .pt file that contains only state_dict using a model class.
    
    This is useful when the .pt file only contains weights (state_dict)
    and you need to provide the model architecture separately.
    
    Args:
        pt_path: Path to .pt file
        model_class: The PyTorch model class (architecture)
        input_shape: Input shape for the model
    
    Returns:
        Path to the converted .onnx file
    """
    
    pt_path = Path(pt_path)
    output_path = pt_path.with_suffix('.onnx')
    
    # 1. Instantiate the model
    try:
        model = model_class()
        model.eval()
    except Exception as e:
        raise ValueError(f"Cannot instantiate model class: {e}")
    
    # 2. Load the state_dict
    try:
        state_dict = torch.load(pt_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Cannot load state_dict: {e}")
    
    # 3. Convert to ONNX
    dummy_input = torch.randn(*input_shape)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    except Exception as e:
        raise ValueError(f"Conversion to ONNX failed: {e}")
    
    return str(output_path)


def detect_input_shape(model) -> tuple:
    """
    Detect the input shape from a PyTorch model.
    """
    try:
        # Try to get input shape from model
        if hasattr(model, 'input_shape'):
            return model.input_shape
        
        # Try to get from model config
        if hasattr(model, 'config') and hasattr(model.config, 'input_shape'):
            return model.config.input_shape
        
        # Try to inspect the model's first layer
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # ✅ FIXED: 'layers' not 'layer'
            first_layer = model.layers[0]
            if hasattr(first_layer, 'input_shape'):
                return first_layer.input_shape
            
    except Exception:
        pass

    print("⚠️ Could not detect input shape, using default: (1, 3, 224, 224)")
    return (1, 3, 224, 224)