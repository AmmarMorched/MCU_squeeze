# mcusqueeze/ingestion/loader.py

from pathlib import Path

from rich import console

from .validator import validate_onnx

def load_model(path: str):
    
    ext = Path(path).suffix.lower()
    
    if ext == '.onnx':
        result = validate_onnx(path)
        
        if not result['valid']:
            raise ValueError(f"Invalid ONNX model: {result['reason']}")
        
        if result['warnings']:
            for w in result['warnings']:
                console.print(f"[yellow]⚠[/] {w}")
        
        return result['model']
    
    elif ext == '.h5':
        # conversion coming next
        pass
    
    else:
        raise ValueError(f"Unsupported format: {ext} — supported: .onnx .h5")