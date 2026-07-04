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


def detect_dynamic_shapes(model):
    """Check if model has dynamic shapes."""
    from mcusqueeze.analysis.graph import get_input_output_shapes
    shapes = get_input_output_shapes(model)
    
    dynamic_dims = []
    for inp in shapes['inputs']:
        if '?' in inp['shape_str']:
            dynamic_dims.append(inp['shape_str'])
    
    return dynamic_dims


def prompt_for_input_size():
    """Ask user for input dimensions."""
    console.print("\n[yellow]⚠️  Model has dynamic input dimensions.[/]")
    console.print("To estimate memory accurately, please provide the input size.")
    console.print("")
    
    height = IntPrompt.ask("Enter image height", default=640)
    width = IntPrompt.ask("Enter image width", default=height)
    
    return height, width


@click.group()
def main():
    """mcusqueeze — Auto-quantize AI models for MCUs."""
    pass


@main.command()
@click.option('--model', required=True, help='Path to model (.h5 or .onnx)')
@click.option('--shapes/--no-shapes', default=True, help='Extract and display tensor shapes')
@click.option('--target', default=DEFAULT_TARGET, help=f'Target MCU. Supported: {", ".join(get_available_targets())} [default: {DEFAULT_TARGET}]')
@click.option('--input-size', default=None, help='Input size for dynamic models (e.g., 224x224). if not provided will prompt')
@click.option('--yes', is_flag=True, help='Auto-accept default dimensions without prompting')
def analyze(model, shapes, target, input_size, yes):
    """Analyze a model without converting it."""
    console.print(Panel(f"[bold cyan]Analyzing:[/] {model}", title="mcusqueeze"))
    if target not in SUPPORTED_TARGETS:
        console.print(f"[red]✗[/] Unsupported target: '{target}'")
        console.print("  Available targets:")
        for t in SUPPORTED_TARGETS.keys():
            console.print(f"    • {t}")
        return
    
    #display target 
    target_info =SUPPORTED_TARGETS[target]
    if target == DEFAULT_TARGET:
        console.print(f"[green]✓[/] Target: {target_info['name']}  [default]")
    else:
        console.print(f"[green]✓[/] Target: {target_info['name']}")

    input_height, input_width = None, None
    if input_size:
        try:
            parts = input_size.lower().split('x')
            input_height= int(parts[0])
            input_width = int(parts[1]) if len(parts) > 1 else input_height
            console.print(f"[green]✓[/] Input size: {input_height}x{input_width} (from flag)")
        except ValueError:
            console.print(f"[red]✗[/] Invalid input-size format. Use: 640x640")
            return

    try: 
        onnx_model = load_model(model, extract_ops=True, extract_shapes=shapes, target=target)
        input_shapes = get_model_input_shapes(onnx_model)
        dynamic_dims = detect_dynamic_dimensions(input_shapes)
        if dynamic_dims and input_height is None and input_width is None and not yes:
            console.print("\n[yellow]⚠️[/] Model has dynamic input dimensions:")
            for shape_info in input_shapes:
                console.print(f"    {shape_info['name']}: {shape_info['shape_str']}")
            default_height, default_width = suggest_default_dimensions(onnx_model, target)
            
            console.print(f"\n[cyan]💡[/] Suggested dimensions based on model analysis:")
            console.print(f"    Height: {default_height}")
            console.print(f"    Width:  {default_width}")
            
            # Ask user
            use_default = Confirm.ask(
                f"\nUse these dimensions for memory estimation?",
                default=True
            )
            
            if use_default:
                input_height, input_width = default_height, default_width
                console.print(f"[green]✓[/] Using dimensions: {input_height}x{input_width}")
            else:
                # User wants to enter custom dimensions
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
        
        # If dimensions provided via flag but model has dynamic dims
        elif dynamic_dims and input_height is not None and input_width is not None:
            console.print(f"[green]✓[/] Using provided dimensions: {input_height}x{input_width}")
        
        # If no dynamic dims and no input size needed
        elif not dynamic_dims:
            console.print("[green]✓[/] Model has static input shapes, no dimensions needed")
        
        # If no dynamic dims but user provided input size anyway
        elif input_height is not None and input_width is not None:
            console.print(f"[green]✓[/] Using provided dimensions: {input_height}x{input_width}")
        
        # If dimensions not provided and yes flag is set, use defaults
        elif dynamic_dims and yes:
            default_height, default_width = suggest_default_dimensions(onnx_model, target)
            input_height, input_width = default_height, default_width
            console.print(f"[yellow]⚠[/] Using default dimensions: {input_height}x{input_width} (--yes flag)")
        
        # Now re-load with the dimensions for memory estimation
        # We need to pass the dimensions to validator
        # This requires updating validator.py to accept dimensions
        
        # For now, reload with dimensions
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
    
    # placeholder for now
    # console.print(f"[green]✓[/] Model loaded: {model}")
    # console.print("[yellow]⚠[/] Analysis not implemented yet")


@main.command()
@click.option('--model',  required=True, help='Path to model (.h5 or .onnx)')
@click.option('--calib',  required=True, help='Path to calibration dataset folder')
@click.option('--target',  default=DEFAULT_TARGET, help=f'Target MCU. Supported: {", ".join(get_available_targets())}  [default: {DEFAULT_TARGET}]')
@click.option('--output', required=True, help='Output folder path')
def run(model, calib, target, output):
    """Run the full quantization pipeline."""
    console.print(Panel("[bold cyan]mcusqueeze v0.1.0[/]", title="Starting Pipeline"))
    # verify target 
    if target not in SUPPORTED_TARGETS:
        console.print(f"[red]✗[/] Unsupported target: '{target}'")
        console.print("  Available targets:")
        for i in get_available_targets():
            console.print(f"    • {t}")
        return
    console.print(f"[green]✓[/] Model:  {model}")
    console.print(f"[green]✓[/] Calib:  {calib}")
    console.print(f"[green]✓[/] Target: {target}")
    console.print(f"[green]✓[/] Output: {output}")
    
    # placeholder for now
    console.print("\n[yellow]Pipeline not implemented yet[/]")