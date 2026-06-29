import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import click
from rich.console import Console
from rich.panel import Panel

from mcusqueeze.ingestion.loader import load_model

console = Console()

@click.group()
def main():
    """mcusqueeze — Auto-quantize AI models for MCUs."""
    pass


@main.command()
@click.option('--model', required=True, help='Path to model (.h5 or .onnx)')
def analyze(model):
    """Analyze a model without converting it."""
    console.print(Panel(f"[bold cyan]Analyzing:[/] {model}", title="mcusqueeze"))
    try: 
        onnx_model = load_model(model)
        console.print(f"[green]✓[/] Model ready for analysis")
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]✗[/] Error: {e}")
    
    # placeholder for now
    # console.print(f"[green]✓[/] Model loaded: {model}")
    # console.print("[yellow]⚠[/] Analysis not implemented yet")


@main.command()
@click.option('--model',  required=True, help='Path to model (.h5 or .onnx)')
@click.option('--calib',  required=True, help='Path to calibration dataset folder')
@click.option('--target', required=True, default='esp32s3', help='Target MCU')
@click.option('--output', required=True, help='Output folder path')
def run(model, calib, target, output):
    """Run the full quantization pipeline."""
    console.print(Panel("[bold cyan]mcusqueeze v0.1.0[/]", title="Starting Pipeline"))
    
    console.print(f"[green]✓[/] Model:  {model}")
    console.print(f"[green]✓[/] Calib:  {calib}")
    console.print(f"[green]✓[/] Target: {target}")
    console.print(f"[green]✓[/] Output: {output}")
    
    # placeholder for now
    console.print("\n[yellow]Pipeline not implemented yet[/]")