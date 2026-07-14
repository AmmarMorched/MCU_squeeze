# mcusqueeze/utils/error_handling.py

"""
Graceful error handling for quantization failures.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional, Dict, Any

console = Console()


class QuantizationError(Exception):
    """Base exception for quantization errors."""
    pass


class InputNameError(QuantizationError):
    """Input name not found in model."""
    
    def __init__(self, provided_name: str, available_names: list):
        self.provided_name = provided_name
        self.available_names = available_names
        super().__init__(
            f"Input name '{provided_name}' not found in model.\n"
            f"Available inputs: {', '.join(available_names)}"
        )


class ShapeMismatchError(QuantizationError):
    """Input shape mismatch."""
    
    def __init__(self, expected: tuple, got: tuple):
        self.expected = expected
        self.got = got
        super().__init__(
            f"Shape mismatch: expected {expected}, got {got}"
        )


class MemoryError(QuantizationError):
    """Out of memory error."""
    
    def __init__(self, suggested_fix: str):
        self.suggested_fix = suggested_fix
        super().__init__(
            f"Out of memory during quantization.\n"
            f"Suggested fix: {suggested_fix}"
        )


class UnsupportedOpsError(QuantizationError):
    """Model contains unsupported operations."""
    
    def __init__(self, ops: list, target: str):
        self.ops = ops
        self.target = target
        super().__init__(
            f"Model contains ops not supported on {target}: {', '.join(ops)}"
        )


def handle_quantization_error(e: Exception, context: Dict[str, Any]) -> bool:
    """
    Handle quantization errors gracefully with user-friendly messages.
    
    Args:
        e: The exception that was raised
        context: Context information (model, target, etc.)
    
    Returns:
        True if user wants to continue, False to abort
    """
    
    console.print("\n[bold red]⚠️ Quantization Failed[/bold red]")
    console.print("-" * 60)
    
    if isinstance(e, InputNameError):
        return _handle_input_name_error(e, context)
    elif isinstance(e, ShapeMismatchError):
        return _handle_shape_mismatch_error(e, context)
    elif isinstance(e, MemoryError):
        return _handle_memory_error(e, context)
    elif isinstance(e, UnsupportedOpsError):
        return _handle_unsupported_ops_error(e, context)
    elif isinstance(e, KeyboardInterrupt):
        console.print("\n[yellow]⏹️ Quantization cancelled by user[/yellow]")
        return False
    else:
        return _handle_unknown_error(e, context)


def _handle_input_name_error(e: InputNameError, context: Dict) -> bool:
    """Handle input name mismatch errors."""
    
    console.print(f"\n[red]❌ Input name mismatch[/red]")
    console.print(f"   You provided: '{e.provided_name}'")
    console.print(f"   Available inputs: {', '.join(e.available_names)}")
    
    # Try to automatically suggest the correct name
    suggested = None
    for name in e.available_names:
        if 'image' in name.lower() or 'input' in name.lower():
            suggested = name
            break
    
    if suggested:
        console.print(f"\n[cyan]💡 Did you mean: '{suggested}'?[/cyan]")
        console.print(f"   Try: --input-name {suggested}")
    
    console.print("\n[yellow]Would you like to retry with the suggested name?[/yellow]")
    from rich.prompt import Confirm
    return Confirm.ask("Retry?", default=True)


def _handle_shape_mismatch_error(e: ShapeMismatchError, context: Dict) -> bool:
    """Handle shape mismatch errors."""
    
    console.print(f"\n[red]❌ Shape mismatch[/red]")
    console.print(f"   Expected: {e.expected}")
    console.print(f"   Got:      {e.got}")
    
    # Check if it's a channel order issue
    if e.expected[1] == 3 and e.got[-1] == 3:
        console.print("\n[cyan]💡 This looks like a channel order issue![/cyan]")
        console.print("   Your model expects NCHW format but you provided NHWC.")
        console.print("   Try: --channel-order NCHW")
    
    console.print("\n[yellow]Would you like to retry with the correct channel order?[/yellow]")
    from rich.prompt import Confirm
    return Confirm.ask("Retry?", default=True)


def _handle_memory_error(e: MemoryError, context: Dict) -> bool:
    """Handle out of memory errors."""
    
    console.print(f"\n[red]❌ Out of memory during quantization[/red]")
    console.print(f"   {e.suggested_fix}")
    
    console.print("\n[cyan]💡 Suggestions to reduce memory usage:[/cyan]")
    console.print("   1. Reduce input size: --input-size 320x320")
    console.print("   2. Reduce batch size: --batch-size 2")
    console.print("   3. Reduce calibration samples: --max-samples 10")
    
    console.print("\n[yellow]Would you like to retry with reduced memory settings?[/yellow]")
    from rich.prompt import Confirm
    return Confirm.ask("Retry?", default=True)


def _handle_unsupported_ops_error(e: UnsupportedOpsError, context: Dict) -> bool:
    """Handle unsupported operations errors."""
    
    console.print(f"\n[red]❌ Unsupported operations on {e.target}[/red]")
    console.print(f"   Unsupported ops: {', '.join(e.ops)}")
    
    # Show which ops are the most problematic
    critical_ops = ['Resize', 'Upsample', 'ConvTranspose', 'LSTM', 'GRU']
    critical_found = [op for op in e.ops if op in critical_ops]
    
    if critical_found:
        console.print(f"\n[yellow]⚠️ Critical ops that cannot be easily replaced:[/yellow]")
        for op in critical_found:
            console.print(f"   • {op}")
    
    console.print("\n[cyan]💡 Suggestions:[/cyan]")
    console.print("   1. Use a different model architecture")
    console.print("   2. Replace unsupported ops with supported alternatives")
    console.print("   3. Choose a different target MCU with better support")
    
    console.print("\n[yellow]Would you like to continue anyway? (Quantization may fail)[/yellow]")
    from rich.prompt import Confirm
    return Confirm.ask("Continue?", default=False)


def _handle_unknown_error(e: Exception, context: Dict) -> bool:
    """Handle unknown errors."""
    
    console.print(f"\n[red]❌ Unknown error during quantization[/red]")
    console.print(f"   Error: {str(e)}")
    
    # Show context
    console.print("\n[cyan]💡 Context:[/cyan]")
    for key, value in context.items():
        console.print(f"   {key}: {value}")
    
    console.print("\n[yellow]Would you like to see the full error traceback?[/yellow]")
    from rich.prompt import Confirm
    if Confirm.ask("Show traceback?", default=False):
        import traceback
        console.print("\n[dim]Full traceback:[/dim]")
        traceback.print_exc()
    
    return False


def format_error_summary(e: Exception) -> str:
    """Format a concise error summary for logging."""
    
    error_type = type(e).__name__
    error_msg = str(e)
    
    # Truncate long messages
    if len(error_msg) > 100:
        error_msg = error_msg[:100] + "..."
    
    return f"[{error_type}] {error_msg}"



# mcusqueeze/utils/error_handling.py

class RetryableError(Exception):
    """Error that can be resolved by retrying with different settings."""
    
    def __init__(self, original_error: Exception, strategies_attempted: list):
        self.original_error = original_error
        self.strategies_attempted = strategies_attempted
        super().__init__(f"All retry strategies failed: {original_error}")


class MemoryError(QuantizationError):
    """Out of memory error with suggestions."""
    
    def __init__(self, original_error: Exception, current_settings: dict):
        self.original_error = original_error
        self.current_settings = current_settings
        suggestions = []
        
        # Build suggestions based on current settings
        if current_settings.get('batch_size', 8) > 1:
            suggestions.append("Reduce batch size: --batch-size 2")
        if current_settings.get('input_shape', (640,640))[0] > 224:
            suggestions.append("Reduce input size: --input-size 320x320")
        if current_settings.get('max_samples', 0) > 10:
            suggestions.append("Reduce samples: --max-samples 10")
        
        super().__init__(
            f"Out of memory during quantization.\n"
            f"Suggestions:\n  " + "\n  ".join(suggestions)
        )





def run_quantization_with_retry(
    quantizer,
    calib,
    input_name,
    height,
    width,
    channels,
    batch_size,
    max_samples,
    channel_order,
    target_options,
):
    """Run quantization with automatic retry on failure."""
    
    # Define fallback strategies
    retry_strategies = [
        # Strategy 1: Try with original settings
        {
            'batch_size': batch_size,
            'input_shape': (height, width, channels),
            'max_samples': max_samples,
            'description': 'original settings',
        },
        # Strategy 2: Reduce batch size by half
        {
            'batch_size': max(batch_size // 2, 1),
            'input_shape': (height, width, channels),
            'max_samples': max_samples,
            'description': 'reduced batch size',
        },
        # Strategy 3: Reduce input size (if > 224)
        {
            'batch_size': max(batch_size // 2, 1),
            'input_shape': (min(height, 224), min(width, 224), channels),
            'max_samples': max_samples,
            'description': 'reduced input size',
        },
        # Strategy 4: Reduce both
        {
            'batch_size': 1,
            'input_shape': (min(height, 224), min(width, 224), channels),
            'max_samples': max_samples,
            'description': 'minimal settings',
        },
        # Strategy 5: Reduce samples
        {
            'batch_size': 1,
            'input_shape': (min(height, 224), min(width, 224), channels),
            'max_samples': min(max_samples or 10, 10),
            'description': 'minimal settings + reduced samples',
        },
    ]
    
    last_error = None
    
    for i, strategy in enumerate(retry_strategies):
        try:
            if i > 0:
                console.print(f"\n[cyan]🔄 Retry attempt {i+1}/{len(retry_strategies)}: {strategy['description']}[/cyan]")
                console.print(f"   Batch size: {strategy['batch_size']}")
                console.print(f"   Input shape: {strategy['input_shape'][0]}x{strategy['input_shape'][1]}")
                console.print(f"   Max samples: {strategy['max_samples'] or 'all'}")
            
            # Run quantization with current strategy
            quantized_path = quantizer.quantize(
                calibration_folder=calib,
                input_name=input_name,
                input_shape=strategy['input_shape'],
                batch_size=strategy['batch_size'],
                max_samples=strategy['max_samples'],
                channel_order=channel_order,
                **target_options,
            )
            
            # Success!
            console.print(f"[green]✓[/] Quantization successful!")
            console.print(f"   Used: {strategy['description']}")
            if i > 0:
                console.print(f"   [green]✅ Automatic retry worked![/green]")
            return quantized_path
            
        except Exception as e:
            last_error = e
            
            # Check if it's a memory error (we can retry)
            is_memory_error = (
                'memory' in str(e).lower() or 
                'killed' in str(e).lower() or
                'allocation' in str(e).lower()
            )
            
            # Check if it's a shape error (we can't fix with retry)
            is_shape_error = (
                'shape' in str(e).lower() or 
                'dimension' in str(e).lower() or
                'incompatible' in str(e).lower()
            )
            
            if is_shape_error:
                console.print(f"[red]✗ Shape error detected: {e}[/red]")
                console.print("[yellow]This cannot be fixed by automatic retry.[/yellow]")
                raise e
                
            if not is_memory_error:
                # Non-memory error - don't retry
                raise e
            
            # Memory error - continue to next strategy
            if i < len(retry_strategies) - 1:
                console.print(f"[yellow]⚠️ Memory issue with current settings[/yellow]")
                console.print(f"   Error: {str(e)[:100]}...")
                console.print(f"   Trying next strategy...")
            else:
                # All strategies failed
                raise e
    
    # If we get here, all retries failed
    raise last_error or RuntimeError("All quantization strategies failed")
