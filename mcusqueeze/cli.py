import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import click
from rich.console import Console
from rich.panel import Panel

from mcusqueeze.ingestion.loader import load_model
from mcusqueeze.exceptions import MCUSqeezeError
from mcusqueeze.targets import SUPPORTED_TARGETS, DEFAULT_TARGET, get_available_targets

console = Console()

@click.group()
def main():
    """mcusqueeze — Auto-quantize AI models for MCUs."""
    pass


@main.command()
@click.option('--model', required=True, help='Path to model (.h5 or .onnx)')
@click.option('--shapes/--no-shapes', default=True, help='Extract and display tensor shapes')
@click.option('--target', default=DEFAULT_TARGET, help=f'Target MCU. Supported: {", ".join(get_available_targets())} [default: {DEFAULT_TARGET}]')
def analyze(model, shapes, target):
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

    try: 
        onnx_model = load_model(model, extract_ops=True, extract_shapes=shapes, target=target)
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