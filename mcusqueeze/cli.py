# mcusqueeze/cli.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from mcusqueeze.ingestion.loader import load_model
from mcusqueeze.exceptions import MCUSqeezeError
from mcusqueeze.targets import SUPPORTED_TARGETS, DEFAULT_TARGET, get_available_targets
from mcusqueeze.analysis.dimensions import detect_dynamic_dimensions, suggest_default_dimensions, get_model_input_shapes

console = Console()


@click.group()
def main():
    """mcusqueeze — Auto-quantize AI models for MCUs."""
    pass


@main.command()
@click.option('--model', required=True, help='Path to model (.h5, .onnx, or .pt)')
@click.option('--shapes/--no-shapes', default=True, help='Extract and display tensor shapes')
@click.option('--target', default=DEFAULT_TARGET, help=f'Target MCU. Supported: {", ".join(get_available_targets())} [default: {DEFAULT_TARGET}]')
@click.option('--input-size', default=None, help='Input size for dynamic models (e.g., 224x224). If not provided will prompt')
@click.option('--yes', is_flag=True, help='Auto-accept default dimensions without prompting')
def analyze(model, shapes, target, input_size, yes):
    """Analyze a model without converting it."""
    console.print(Panel(f"[bold cyan]Analyzing:[/] {model}", title="mcusqueeze"))
    
    # Validate target
    if target not in SUPPORTED_TARGETS:
        console.print(f"[red]✗[/] Unsupported target: '{target}'")
        console.print("  Available targets:")
        for t in SUPPORTED_TARGETS.keys():
            console.print(f"    • {t}")
        return
    
    # Display target
    target_info = SUPPORTED_TARGETS[target]
    if target == DEFAULT_TARGET:
        console.print(f"[green]✓[/] Target: {target_info['name']}  [default]")
    else:
        console.print(f"[green]✓[/] Target: {target_info['name']}")

    # Parse input size if provided
    input_height, input_width = None, None
    if input_size:
        try:
            parts = input_size.lower().split('x')
            input_height = int(parts[0])
            input_width = int(parts[1]) if len(parts) > 1 else input_height
            console.print(f"[green]✓[/] Input size: {input_height}x{input_width} (from flag)")
        except ValueError:
            console.print(f"[red]✗[/] Invalid input-size format. Use: 640x640")
            return

    try:
        # First load to check for dynamic dimensions
        onnx_model = load_model(
            model,
            extract_ops=True,
            extract_shapes=shapes,
            target=target
        )
        
        # Check for dynamic dimensions
        input_shapes = get_model_input_shapes(onnx_model)
        dynamic_dims = detect_dynamic_dimensions(input_shapes)
        
        # Handle dynamic dimensions
        if dynamic_dims and input_height is None and input_width is None and not yes:
            console.print("\n[yellow]⚠️[/] Model has dynamic input dimensions:")
            for shape_info in input_shapes:
                console.print(f"    {shape_info['name']}: {shape_info['shape_str']}")
            
            default_height, default_width = suggest_default_dimensions(onnx_model, target)
            
            console.print(f"\n[cyan]💡[/] Suggested dimensions based on model analysis:")
            console.print(f"    Height: {default_height}")
            console.print(f"    Width:  {default_width}")
            
            use_default = Confirm.ask(
                f"\nUse these dimensions for memory estimation?",
                default=True
            )
            
            if use_default:
                input_height, input_width = default_height, default_width
                console.print(f"[green]✓[/] Using dimensions: {input_height}x{input_width}")
            else:
                while True:
                    try:
                        input_height = Prompt.ask(
                            "[cyan]Enter height[/]",
                            default=str(default_height)
                        )
                        input_width = Prompt.ask(
                            "[cyan]Enter width[/]",
                            default=str(default_width)
                        )
                        input_height = int(input_height)
                        input_width = int(input_width)
                        if input_height > 0 and input_width > 0:
                            break
                        console.print("[red]✗[/] Dimensions must be positive integers")
                    except ValueError:
                        console.print("[red]✗[/] Please enter valid integers")
                
                console.print(f"[green]✓[/] Using custom dimensions: {input_height}x{input_width}")
        
        elif dynamic_dims and input_height is not None and input_width is not None:
            console.print(f"[green]✓[/] Using provided dimensions: {input_height}x{input_width}")
        
        elif not dynamic_dims:
            console.print("[green]✓[/] Model has static input shapes, no dimensions needed")
        
        elif dynamic_dims and yes:
            default_height, default_width = suggest_default_dimensions(onnx_model, target)
            input_height, input_width = default_height, default_width
            console.print(f"[yellow]⚠[/] Using default dimensions: {input_height}x{input_width} (--yes flag)")
        
        # Re-load with dimensions for memory estimation
        onnx_model = load_model(
            model,
            extract_ops=True,
            extract_shapes=shapes,
            target=target,
            input_height=input_height,
            input_width=input_width
        )
        
        console.print(f"[green]✓[/] Model ready for analysis")
        
    except MCUSqeezeError as e:
        console.print(f"[red]✗[/] Error: {e}")


@main.command()
@click.option('--model', required=True, help='Path to model (.h5, .onnx, or .pt)')
@click.option('--calib', required=True, help='Path to calibration dataset folder')
@click.option('--target', default=DEFAULT_TARGET, help=f'Target MCU. Supported: {", ".join(get_available_targets())} [default: {DEFAULT_TARGET}]')
@click.option('--output', required=True, help='Output folder path for quantized model')
@click.option('--batch-size', default=8, help='Batch size for calibration')
@click.option('--max-samples', default=None, type=int, help='Max samples to use from calibration dataset')
@click.option('--input-size', default=None, help='Input size for dynamic models (e.g., 224x224)')
@click.option('--yes', is_flag=True, help='Auto-accept default dimensions without prompting')
def run(model, calib, target, output, batch_size, max_samples, input_size, yes):
    """
    Run the full quantization pipeline.
    
    This takes a float32 model and produces an int8 quantized model
    optimized for the target MCU.
    """
    console.print(Panel("[bold cyan]mcusqueeze v0.1.0[/]", title="Quantization Pipeline"))
    
    # Validate target
    if target not in SUPPORTED_TARGETS:
        console.print(f"[red]✗[/] Unsupported target: '{target}'")
        console.print("  Available targets:")
        for t in get_available_targets():
            console.print(f"    • {t}")
        return
    
    target_info = SUPPORTED_TARGETS[target]
    console.print(f"[green]✓[/] Target: {target_info['name']}")
    console.print(f"[green]✓[/] Model:  {model}")
    console.print(f"[green]✓[/] Calib:  {calib}")
    console.print(f"[green]✓[/] Output: {output}")
    
    # Parse input size if provided
    input_height, input_width = None, None
    if input_size:
        try:
            parts = input_size.lower().split('x')
            input_height = int(parts[0])
            input_width = int(parts[1]) if len(parts) > 1 else input_height
            console.print(f"[green]✓[/] Input size: {input_height}x{input_width} (from flag)")
        except ValueError:
            console.print(f"[red]✗[/] Invalid input-size format. Use: 640x640")
            return
    
    try:
        # 1. Load the model to check for dynamic dimensions
        console.print("\n[cyan]→[/] Loading model...")
        onnx_model = load_model(
            model,
            extract_ops=True,
            extract_shapes=True,
            target=target
        )
        console.print(f"[green]✓[/] Model loaded successfully")
        
        # 2. Get input shape from model
        from mcusqueeze.analysis.graph import get_input_output_shapes
        shapes = get_input_output_shapes(onnx_model)
        
        if not shapes['inputs']:
            console.print("[red]✗[/] Could not determine input shape")
            return
        
        input_name = shapes['inputs'][0]['name']
        input_shape = shapes['inputs'][0]['shape']
        
        # Replace dynamic dimensions with 1 (batch)
        input_shape_fixed = [1 if d == -1 else d for d in input_shape]
        height, width, channels = 224, 224, 3
        # Determine height, width, channels
        if len(input_shape) == 4:
            has_dynamic = any(d == -1 for d in input_shape)
            if input_height is None or input_width is None:
                # Try to get from model if not dynamic
                has_dynamic = any(d == -1 for d in input_shape)
                if not has_dynamic:
                    # Use the model's shape
                    if input_shape[1] in [1, 3, 4]:  # Likely (N, C, H, W)
                        height, width, channels = input_shape[2], input_shape[3], input_shape[1]
                    else:  # Likely (N, H, W, C)
                        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
                    console.print(f"📐 Detected input: ({input_name}) {height}x{width}x{channels} (static)")
                else:
                    # Need user input
                    console.print("\n[yellow]⚠️[/] Model has dynamic input shape:")
                    console.print(f"    {input_name}: {shapes['inputs'][0]['shape_str']}")
                    
                    if not yes:
                        default_height, default_width = 640, 640
                        console.print(f"\n[cyan]💡[/] Suggested: {default_height}x{default_width}")
                        use_default = Confirm.ask(
                            f"\nUse these dimensions for quantization?",
                            default=True
                        )
                        if use_default:
                            input_height, input_width = default_height, default_width
                        else:
                            input_height = IntPrompt.ask("Enter height", default=default_height)
                            input_width = IntPrompt.ask("Enter width", default=input_height)
                        console.print(f"[green]✓[/] Using: {input_height}x{input_width}")
                    else:
                        input_height, input_width = 640, 640
                        console.print(f"[yellow]⚠[/] Using default: {input_height}x{input_width} (--yes)")
        else:
            # Not a 4D tensor (image), use defaults
            height, width, channels = 224, 224, 3
            console.print(f"📐 Using default input: {height}x{width}x{channels}")
        
        console.print(f"📐 Input shape: ({input_name}) {height}x{width}x{channels}")
        
        # 3. Load calibration dataset
        console.print("\n[cyan]→[/] Loading calibration dataset...")
        from mcusqueeze.quantization.calibration import get_calibration_data
        
        # Use provided dimensions or detected ones
        calib_height = input_height if input_height else height
        calib_width = input_width if input_width else width
        
        # Collect calibration data
        calib_batches = list(get_calibration_data(
            folder_path=calib,
            input_shape=(calib_height, calib_width, channels),
            batch_size=batch_size,
            max_samples=max_samples
        ))
        
        total_samples = sum(len(batch) for batch in calib_batches)
        console.print(f"[green]✓[/] Loaded {total_samples} calibration images")
        
        # 4. Run calibration
        console.print("\n[cyan]→[/] Running calibration...")
        
        # TODO: Next step - run inference and collect statistics
        console.print("[yellow]⚠[/] Calibration inference not yet implemented")
        console.print("    This is the next step in the quantization pipeline")
        
        # 5. Quantize model
        console.print("\n[cyan]→[/] Quantizing model...")
        console.print("[yellow]⚠[/] Quantization not yet implemented")
        
        # 6. Export quantized model
        console.print("\n[cyan]→[/] Exporting quantized model...")
        console.print("[yellow]⚠[/] Export not yet implemented")
        
        console.print("\n[yellow]Quantization pipeline under development[/]")
        console.print("Current step: Calibration dataset loader (working)")
        console.print("Next steps: Inference → Statistics → Quantization → Export")
        
    except MCUSqeezeError as e:
        console.print(f"[red]✗[/] Error: {e}")
    except Exception as e:
        console.print(f"[red]✗[/] Unexpected error: {e}")


if __name__ == '__main__':
    main()