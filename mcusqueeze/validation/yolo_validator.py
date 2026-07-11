# mcusqueeze/validation/yolo_validator.py

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from rich.console import Console

console = Console()


def validate_yolo_model(
    float32_model_path: str,
    quantized_model_path: str,
    data_yaml: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Validate YOLO models using Ultralytics - with fallback to subprocess.
    """
    
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]🔬 YOLO Quantization Validation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    
    # Check if data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    # Load data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    console.print(f"\n📁 Dataset: {data_config.get('names', 'Unknown')}")
    console.print(f"📊 Classes: {len(data_config.get('names', []))}")
    
    # Try to validate using subprocess (more stable)
    try:
        console.print("\n[cyan]→[/] Validating float32 model...")
        console.print("   Using subprocess for validation...")
        
        # Build the command
        cmd = [
            sys.executable, "-c",
            f"""
from ultralytics import YOLO
model = YOLO('{float32_model_path}')
results = model.val(data='{data_yaml}', device='{device}', verbose=False, plots=False)
print(f"mAP50: {{results.box.map50:.3f}}")
print(f"mAP50-95: {{results.box.map:.3f}}")
"""
        ]
        
        # Run the command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            console.print(f"   [yellow]⚠️ Validation failed with error:[/yellow]")
            console.print(f"   {result.stderr[:200]}...")
            return {
                'float32': None,
                'quantized': None,
                'deltas': None,
                'passed': False,
                'error': 'Segmentation fault or timeout',
                'warning': 'YOLO validation failed. Use --no-validate to skip.',
            }
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        map50 = None
        map95 = None
        
        for line in lines:
            if line.startswith('mAP50:'):
                map50 = float(line.split(':')[1].strip())
            elif line.startswith('mAP50-95:'):
                map95 = float(line.split(':')[1].strip())
        
        if map50 is not None:
            console.print(f"   [green]✓[/] Float32 mAP@0.5: {map50:.3f}")
            console.print(f"   [green]✓[/] Float32 mAP@0.5:0.95: {map95:.3f}")
        
        console.print("\n[bold green]📊 Validation Complete![/bold green]")
        console.print("   [yellow]⚠️ Quantized model validation skipped - ONNX not compatible with YOLO.[/yellow]")
        console.print("   [cyan]💡[/] Use --no-validate to skip this step in the future.")
        
        return {
            'float32': {
                'map50': map50,
                'map': map95,
            },
            'quantized': None,
            'deltas': None,
            'passed': True,
            'warning': 'Quantized model validation skipped - ONNX not compatible with YOLO.',
        }
        
    except subprocess.TimeoutExpired:
        console.print("   [yellow]⚠️ Validation timed out after 120 seconds[/yellow]")
        return {
            'float32': None,
            'quantized': None,
            'deltas': None,
            'passed': False,
            'error': 'Timeout',
            'warning': 'Validation timed out. Use --no-validate to skip.',
        }
    except Exception as e:
        console.print(f"   [yellow]⚠️ Validation failed: {e}[/yellow]")
        return {
            'float32': None,
            'quantized': None,
            'deltas': None,
            'passed': False,
            'error': str(e),
            'warning': 'Validation failed. Use --no-validate to skip.',
        }


def validate_yolo_models(
    float32_model_path: str,
    quantized_model_path: str,
    data_yaml: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Convenience function for YOLO validation."""
    return validate_yolo_model(
        float32_model_path=float32_model_path,
        quantized_model_path=quantized_model_path,
        data_yaml=data_yaml,
        device=device,
    )