import click
from rich.console import Console
from rich.panel import Panel

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
    
    # placeholder for now
    console.print(f"[green]✓[/] Model loaded: {model}")
    console.print("[yellow]⚠[/] Analysis not implemented yet")


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