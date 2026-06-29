from pathlib import Path
from rich.console import Console
from .validator import validate_onnx
from .h5_converter import convert_h5_to_onnx

console = Console()
    
def load_model(path: str):

    path = Path(path)
    ext = Path(path).suffix.lower()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if ext == '.onnx':
        console.print(f"[cyan]→[/] ONNX model detected — validating...")
        result = validate_onnx(path)
        
        if not result['valid']:
            raise ValueError(f"Invalid ONNX model: {result['reason']}")
        
        for w in result.get('warnings', []):
            console.print(f"[yellow]⚠[/] {w}")

        console.print(f"[green]✓[/] ONNX validated (opset {result['opset']})")
        return result['model']
    
    elif ext in ['.h5', '.keras']:
        console.print(f"[cyan]→[/] Keras model detected — converting to ONNX...")

        try:
            onnx_path = convert_h5_to_onnx(path)
            console.print(f"[green]✓[/] Conversion successful: {onnx_path}")
        except ValueError as e:
            raise ValueError(f"Conversion failed: {e}")
        
        result = validate_onnx(onnx_path)
        
        if not result['valid']:
            raise ValueError(f"Converted ONNX model is invalid: {result['reason']}")
        console.print(f"[green]✓[/] Converted ONNX validated (opset {result['opset']})")
        return result['model']
    
    else:
        raise ValueError(f"Unsupported format: {ext} — supported: .onnx .h5")