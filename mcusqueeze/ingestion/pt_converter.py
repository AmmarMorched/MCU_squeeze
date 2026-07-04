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
            # If this works, it's a YOLO model
            YOLO(str(pt_path))
            return True
        except ImportError:
            # Ultralytics not installed, try to check file contents
            pass
        except Exception:
            # Not a YOLO model
            pass
        
        # Alternative: Check file contents
        try:
            # Load with weights_only=False to inspect
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            
            # Check for YOLO-specific keys or patterns
            if isinstance(data, dict):
                # YOLO models often have these keys
                yolo_keys = ['model', 'names', 'nc', 'stride', 'pt']
                if any(key in data for key in yolo_keys):
                    return True
                
                # Check if it has a 'model' key that contains YOLO-like structure
                if 'model' in data and hasattr(data['model'], 'model'):
                    return True
                    
            # Check if it's a model object with YOLO attributes
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
    
    # Default output path
    if output_path is None:
        output_path = pt_path.with_suffix('.onnx')
    
    output_path = Path(output_path)

    # === STEP 1: AUTO-DETECT YOLO ===
    if is_yolo_model(pt_path):
        print("🟢 YOLO model detected - using Ultralytics export")
        
        try:
            from ultralytics import YOLO
            
            # Load the YOLO model
            print(f"Loading YOLO model from {pt_path}...")
            model = YOLO(str(pt_path))
            
            # Export to ONNX
            print(f"Exporting to ONNX at {output_path}...")
            model.export(
                format='onnx',
                imgsz=640,  # Default YOLO size
                opset=17,
                simplify=True,
                dynamic=True,
                device='cpu'
            )
            
            # The export creates the file with default name
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
    try:
        # Try loading with weights_only=True first (safer)
        model = torch.load(pt_path, map_location='cpu', weights_only=True)
        print("✅ Model loaded with weights_only=True")
        # If successful, it's likely a state_dict or a simple model
    except Exception:
        # If it fails, it's probably a full model object
        print(f"⚠️  Warning: Falling back to weights_only=False. This is only safe for files from trusted sources.")
        try:
            model = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Cannot load .pt model: {e}")
        
        # If it's a state_dict, wrap it in a model
        if isinstance(model, dict):
            # If it's just a state_dict, you need to know the model architecture
            # This is a limitation - you need the model class
            raise ValueError(
                f"'{pt_path}' contains only model weights (state_dict).\n"
                f"Please provide a complete model object or use a model class with load_state_dict().\n"
                f"Tips:\n"
                f"  - For YOLO models, install ultralytics: pip install ultralytics\n"
                f"  - For other models, load with the model class first"
                )
        
        # Set to eval mode
        model.eval()

        try:
        #create a dummy input
            dummy_input = torch.randn(*input_shape)
            print(f"✅ Dummy input created with shape {input_shape}")
        except Exception as e:
            raise ValueError(f"Cannot create dummy input: {e}")
        

        # Export to ONNX
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
            raise ValueError(f"Cannot load .pt model: {e}")
    
    # 2. Create dummy input for tracing
    try:
        dummy_input = torch.randn(*input_shape)
    except Exception as e:
        raise ValueError(f"Cannot create dummy input: {e}")
    
    # 3. Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,  # or latest supported
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # variable batch size
                'output': {0: 'batch_size'}
            }
        )
    except Exception as e:
        raise ValueError(f"Conversion to ONNX failed: {e}")
    
    return str(output_path)


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
        #try to get input shape from model 
        if hasattr(model,'input_shape'):
            return model.input_shape
        
        #try to get from model config
        if hasattr(model,'config') and hasattr(model.config,'input_shape'):
            return model.config.input_shape
        
        #try to inspect the model first layer 
        if hasattr(model, 'layers')and len(model.layers) > 0:
            first_layer = model.layer[0]
            if hasattr(first_layer, 'input_shape'):
                return first_layer.input_shape
            
    except Exception:
        pass

    print("⚠️ Could not detect input shape, using default: (1, 3, 224, 224)")
    return (1, 3, 224, 224)