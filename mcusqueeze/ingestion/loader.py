from pathlib import Path
from rich.console import Console
from .validator import validate_onnx
from .h5_converter import convert_h5_to_onnx
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
    
def load_model(path: str):

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
            result = validate_onnx(str(path))
        except PermissionError:
            raise InvalidONNXError(path)
        
        if not result['valid']:
            raise InvalidONNXError(result['result'])
        
        for w in result.get('warnings', []):
            console.print(f"[yellow]⚠[/] {w}")

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
        
        result = validate_onnx(onnx_path)
        
        if not result['valid']:
            raise InvalidONNXError(result['result'])
        console.print(f"[green]✓[/] Converted ONNX validated (opset {result['opset']})")
        return result['model']
    
    else:
        raise UnsupportedFormatError(ext, path.name)