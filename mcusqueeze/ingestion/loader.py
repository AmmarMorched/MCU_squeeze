from pathlib import Path
from rich.console import Console
from .validator import validate_onnx
from .h5_converter import convert_h5_to_onnx
from mcusqueeze.analysis.graph import get_model_summary, get_input_output_summary
from mcusqueeze.analysis.compatibility import get_compatibility_summary
from mcusqueeze.exceptions import (
    FileNotFoundError,
    EmptyFileError,
    DirectoryGivenError,
    PermissionError,
    UnsupportedFormatError,
    ConversionError,
    InvalidONNXError,
)

console = Console()
    
def load_model(path: str, extract_ops:bool = False, extract_shapes:bool=False, target:str='esp32s3'):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)
    
    if path.is_dir():
        raise DirectoryGivenError(path)
    
    if path.stat().st_size == 0:
        raise EmptyFileError(path)
    

    ext = Path(path).suffix.lower()

    if ext == '.onnx':
        console.print(f"[cyan]→[/] ONNX model detected — validating...")
        
        try:
            result = validate_onnx(str(path), extract_ops=extract_ops, extract_shapes=extract_shapes, target=target)
        except PermissionError:
            raise InvalidONNXError(path)
        
        if not result['valid']:
            raise InvalidONNXError(result['reason'])
        
        for w in result.get('warnings', []):
            console.print(f"[yellow]⚠[/] {w}")

        if extract_shapes and result.get('io_shapes'):
            console.print("[cyan]→[/] Model Inputs/Outputs:")
            console.print(get_input_output_summary(result['model']))


        if extract_ops and result.get('op_analysis'):
            console.print("[cyan]→[/] Extracted operations:")
            console.print(get_model_summary(result['model']))

        if extract_shapes and result.get('model_size'):
            console.print("[cyan]→[/] Model Size:")
            from mcusqueeze.analysis.memory import get_memory_summary
            console.print(get_memory_summary(result['model']))


        if extract_shapes and result.get('comptability'):
            console.print("[cyan]→[/] Target Compatibility:")
            console.print(get_compatibility_summary(result['compatibility']))

        console.print(f"[green]✓[/] ONNX validated (opset {result['opset']})")
        return result['model']
    
    elif ext in ['.h5', '.keras']:
        console.print(f"[cyan]→[/] Keras model detected — converting to ONNX...")

        try:
            onnx_path = convert_h5_to_onnx(path)
        except PermissionError:
            raise PermissionError(path)
        except ValueError as e:
            raise ConversionError(str(e))
        
        result = validate_onnx(onnx_path, extract_ops=extract_ops, extract_shapes=extract_shapes, target=target)
        
        if not result['valid']:
            raise InvalidONNXError(result['reason'])
        
        #print odel summary with shapes
        if extract_shapes and result.get('io_shapes'):
            console.print("[cyan]→[/] Model Inputs/Outputs:")
            console.print(get_input_output_summary(result['model']))

        if extract_ops and result.get('op_analysis'):
            console.print("[cyan]→[/] Extracted operations:")
            console.print(get_model_summary(result['model']))

        if extract_shapes and result.get('compatibility'):
            console.print("[cyan]→[/] Target Compatibility:")
            console.print(get_compatibility_summary(result['compatibility']))

        console.print(f"[green]✓[/] Converted ONNX validated (opset {result['opset']})")
        return result['model']
    
    else:
        raise UnsupportedFormatError(ext, path.name)