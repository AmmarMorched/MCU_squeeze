# mcusqueeze/cli.py

import os
from pathlib import Path
import traceback

from mcusqueeze.analysis.graph import get_input_output_shapes
from mcusqueeze.quantization.ptq import PTQ, get_quantization_options_for_target
from mcusqueeze.validation.validator import validate_quantization
from mcusqueeze.validation.yolo_validator import validate_yolo_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from mcusqueeze.quantization.calibration import get_calibration_data
from mcusqueeze.ingestion.loader import load_model
from mcusqueeze.exceptions import MCUSqeezeError
from mcusqueeze.targets import SUPPORTED_TARGETS, DEFAULT_TARGET, get_available_targets
from mcusqueeze.analysis.dimensions import detect_dynamic_dimensions, suggest_default_dimensions, get_model_input_shapes

console = Console()


def detect_channel_order(input_shape) -> str:
    """
    Detect if model expects NCHW or NHWC format.
    
    Args:
        input_shape: Shape tuple from model
    
    Returns:
        'NCHW' for PyTorch/ONNX format, 'NHWC' for TensorFlow format
    """
    if len(input_shape) == 4:
        # Check if channels dimension is 3 or 4 (likely channels-first)
        if input_shape[1] in [1, 3, 4]:
            return 'NCHW'  # PyTorch format
        elif input_shape[-1] in [1, 3, 4]:
            return 'NHWC'  # TensorFlow format
    return 'NHWC'  # Default to NHWC


def get_input_dimensions(input_shape, input_name, shape_str, input_height=None, input_width=None, yes=False):
    """
    Get input dimensions from model or user input.
    
    Returns:
        (height, width, channels)
    """
    height, width, channels = 224, 224, 3
    
    if len(input_shape) != 4:
        console.print(f"📐 Using default input: {height}x{width}x{channels}")
        return height, width, channels
    
    has_dynamic = any(d == -1 for d in input_shape)
    
    if not has_dynamic:
        # Static shape - detect format
        if input_shape[1] in [1, 3, 4]:  # NCHW
            height, width, channels = input_shape[2], input_shape[3], input_shape[1]
        else:  # NHWC
            height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        console.print(f"📐 Detected input: ({input_name}) {height}x{width}x{channels} (static)")
        return height, width, channels
    
    # Dynamic shape - need user input
    console.print("\n[yellow]⚠️[/] Model has dynamic input shape:")
    console.print(f"    {input_name}: {shape_str}")
    
    if input_height is not None and input_width is not None:
        height, width = input_height, input_width
        console.print(f"[green]✓[/] Using provided dimensions: {height}x{width}")
        return height, width, channels
    
    if not yes:
        default_height, default_width = 640, 640
        console.print(f"\n[cyan]💡[/] Suggested: {default_height}x{default_width}")
        
        if Confirm.ask(f"\nUse these dimensions for quantization?", default=True):
            height, width = default_height, default_width
        else:
            height = IntPrompt.ask("Enter height", default=default_height)
            width = IntPrompt.ask("Enter width", default=height)
        console.print(f"[green]✓[/] Using: {height}x{width}")
    else:
        height, width = 640, 640
        console.print(f"[yellow]⚠[/] Using default: {height}x{width} (--yes)")
    
    # Detect channels
    for d in input_shape:
        if d not in [-1, 0] and d not in [height, width]:
            if d in [1, 3, 4]:
                channels = d
                break
    
    return height, width, channels


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
@click.option('--no-validate', is_flag=True, help='Skip validation step')
@click.option('--data-yaml', default=None, help='Path to data.yaml for YOLO validation')
def run(model, calib, target, output, batch_size, max_samples, input_size, yes, no_validate, data_yaml):
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
    
    onnx_path = None
    quantized_path = None
    height = width = channels = 224
    input_name = 'input'
    
    try:
        # 1. Load the model
        console.print("\n[cyan]→[/] Loading model...")
        onnx_model, onnx_path = load_model(
            model,
            extract_ops=True,
            extract_shapes=True,
            target=target
        )
        console.print(f"[green]✓[/] Model loaded successfully")
        console.print(f"   ONNX file: {onnx_path}")
        
        # 2. Get input shape from model
        shapes = get_input_output_shapes(onnx_model)
        
        if not shapes['inputs']:
            console.print("[red]✗[/] Could not determine input shape")
            return
        
        input_name = shapes['inputs'][0]['name']
        input_shape = shapes['inputs'][0]['shape']
        
        # Detect channel order and input dimensions
        channel_order = detect_channel_order(input_shape)
        console.print(f"📐 Channel order: {channel_order}")
        
        # Get input dimensions
        height, width, channels = get_input_dimensions(
            input_shape=input_shape,
            input_name=input_name,
            shape_str=shapes['inputs'][0]['shape_str'],
            input_height=input_height,
            input_width=input_width,
            yes=yes
        )
        
        console.print(f"📐 Input shape: ({input_name}) {height}x{width}x{channels}")
        
        # 3. Load calibration dataset (count samples)
        console.print("\n[cyan]→[/] Loading calibration dataset...")
        
        calib_height = input_height if input_height else height
        calib_width = input_width if input_width else width
        
        console.print("   Counting calibration samples...")
        sample_count = 0
        for batch in get_calibration_data(
            folder_path=calib,
            input_shape=(calib_height, calib_width, channels),
            batch_size=batch_size,
            max_samples=max_samples,
            channel_order=channel_order
        ):
            sample_count += len(batch)
        
        console.print(f"[green]✓[/] Found {sample_count} calibration images")
        
        # 4. Run calibration inference
        console.print("\n[cyan]→[/] Running calibration inference...")
        
        try:
            import onnxruntime as ort
            import numpy as np
            
            # Load the ONNX model
            console.print("   Loading ONNX model for inference...")
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            # Get input and output names
            input_name = session.get_inputs()[0].name
            output_names = [out.name for out in session.get_outputs()]
            
            console.print(f"   Input: {input_name}")
            console.print(f"   Outputs: {len(output_names)} tensors")
            
            # Initialize statistics collector
            layer_stats = {}
            batch_count = 0
            
            console.print("   Running inference on calibration data...")
            
            # Stream calibration data directly
            for batch in get_calibration_data(
                folder_path=calib,
                input_shape=(calib_height, calib_width, channels),
                batch_size=batch_size,
                max_samples=max_samples,
                channel_order=channel_order
            ):
                batch_count += 1
                
                # Ensure float32
                if batch.dtype != np.float32:
                    batch = batch.astype(np.float32)
                
                if batch_count % 5 == 0 or batch_count == 1:
                    console.print(f"      Batch {batch_count}: shape {batch.shape}, dtype: {batch.dtype}")
                
                # Run inference
                try:
                    outputs = session.run(None, {input_name: batch})
                except Exception as e:
                    console.print(f"      [red]✗ Inference failed for batch {batch_count}: {e}[/]")
                    continue
                
                # Collect statistics for each output
                for i, out_tensor in enumerate(outputs):
                    out_name = output_names[i] if i < len(output_names) else f"output_{i}"
                    
                    if out_name not in layer_stats:
                        layer_stats[out_name] = {
                            'min': [],
                            'max': [],
                            'mean': [],
                            'std': []
                        }
                    
                    layer_stats[out_name]['min'].append(np.min(out_tensor))
                    layer_stats[out_name]['max'].append(np.max(out_tensor))
                    layer_stats[out_name]['mean'].append(np.mean(out_tensor))
                    layer_stats[out_name]['std'].append(np.std(out_tensor))
            
            if batch_count == 0:
                console.print("[red]✗ No batches were processed![/]")
                console.print("   Check your calibration folder and batch size.")
                return
            
            console.print(f"[green]✓[/] Processed {batch_count} batches")
            
            # Calculate quantization parameters
            console.print("\n[cyan]→[/] Calculating quantization parameters...")
            
            quantization_params = {}
            for layer_name, stats in layer_stats.items():
                min_val = min(stats['min'])
                max_val = max(stats['max'])
                mean_val = np.mean(stats['mean'])
                std_val = np.mean(stats['std'])
                
                if max_val > min_val:
                    scale = (max_val - min_val) / 255.0
                    zero_point = -min_val / scale
                else:
                    scale = 1.0
                    zero_point = 0.0
                
                quantization_params[layer_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'scale': scale,
                    'zero_point': zero_point,
                    'range': max_val - min_val,
                }
            
            console.print(f"[green]✓[/] Calculated parameters for {len(quantization_params)} layers")
            
            # Show statistics
            console.print("\n[cyan]→[/] Layer Statistics Summary:")
            console.print("   " + "-" * 50)
            
            sorted_layers = sorted(
                quantization_params.items(),
                key=lambda x: x[1]['range'],
                reverse=True
            )[:5]
            
            for layer_name, params in sorted_layers:
                console.print(f"   {layer_name}:")
                console.print(f"      min: {params['min']:.4f}, max: {params['max']:.4f}")
                console.print(f"      scale: {params['scale']:.6f}, zero_point: {params['zero_point']:.2f}")
            
            console.print("   " + "-" * 50)
            console.print("[green]✓[/] Calibration inference complete!")
            console.print(f"   Total layers analyzed: {len(quantization_params)}")
            
        except ImportError as e:
            console.print(f"[red]✗[/] Import error: {e}")
            console.print("   Make sure ONNX Runtime is installed:")
            console.print("   pip install onnxruntime")
            return
        except Exception as e:
            console.print(f"[red]✗[/] Calibration inference failed: {e}")
            traceback.print_exc()
            return
        
        # 5. Quantize model
        console.print("\n[cyan]→[/] Quantizing model...")
        
        try:
            # Get target-specific quantization options
            target_options = get_quantization_options_for_target(target)
            
            output_path = Path(output) / "quantized_model.onnx"
            
            # Create quantizer
            quantizer = PTQ(
                model_path=onnx_path,
                output_path=output_path,
                target=target,
            )
            
            # Run quantization
            quantized_path = quantizer.quantize(
                calibration_folder=calib,
                input_name=input_name,
                input_shape=(height, width, channels),
                batch_size=batch_size,
                max_samples=max_samples,
                channel_order=channel_order,
                **target_options,
            )
            
            console.print(f"[green]✓[/] Quantized model saved to: {quantized_path}")
            
        except ImportError as e:
            console.print("[red]✗[/] ONNX Runtime quantization not available")
            console.print("   Install required packages:")
            console.print("   pip install onnxruntime-quantization")
            console.print(f"   Error: {e}")
            quantized_path = None
        except Exception as e:
            console.print(f"[red]✗[/] Quantization failed: {e}")
            traceback.print_exc()
            quantized_path = None
        
        # 6. Validate quantized model
        if not no_validate and quantized_path:
            await_run_validation(onnx_path, quantized_path, calib, input_name, height, width, channels, batch_size, max_samples, channel_order, data_yaml)
        else:
            if no_validate:
                console.print("[yellow]⚠️[/] Validation skipped (--no-validate flag)")
            else:
                console.print("[yellow]⚠️[/] Quantization failed, skipping validation...")
    
        # 7. Export quantized model
        console.print("\n[cyan]→[/] Exporting quantized model...")
        console.print("[yellow]⚠[/] Export not yet implemented")
        
        console.print("\n[yellow]Quantization pipeline under development[/]")
        console.print("Current step: Quantization (working)")
        console.print("Next steps: Export → Deployment")
        
    except MCUSqeezeError as e:
        console.print(f"[red]✗[/] Error: {e}")
    except Exception as e:
        console.print(f"[red]✗[/] Unexpected error: {e}")
        traceback.print_exc()


def await_run_validation(onnx_path, quantized_path, calib, input_name, height, width, channels, batch_size, max_samples, channel_order, data_yaml):
    """Handle model validation with support for YOLO."""
    console.print("\n[cyan]→[/] Validating quantized model...")
    
    if not Path(onnx_path).exists():
        console.print(f"[red]✗[/] Float32 model not found: {onnx_path}")
        console.print("   Skipping validation...")
        return
    elif not Path(quantized_path).exists():
        console.print(f"[red]✗[/] Quantized model not found: {quantized_path}")
        console.print("   Skipping validation...")
        return
    
    try:
        # Try YOLO validation if data.yaml is provided
        if data_yaml and Path(data_yaml).exists():
            console.print(f"   Using YOLO validation with data.yaml: {data_yaml}")
            try:
                validation_results = validate_yolo_model(
                    float32_model_path=onnx_path,
                    quantized_model_path=quantized_path,
                    data_yaml=data_yaml,
                    device='cpu',
                )
                
                if validation_results:
                    float32_map = validation_results['float32']['map50']
                    quantized_map = validation_results['quantized']['map50']
                    delta = validation_results['deltas']['map50']
                    
                    console.print(f"[green]✓[/] YOLO validation complete!")
                    console.print(f"   Float32 mAP@0.5:   {float32_map:.3f}")
                    console.print(f"   Quantized mAP@0.5: {quantized_map:.3f}")
                    console.print(f"   Delta:             {delta:.3f} ({delta*100:.1f}%)")
                    
                    if validation_results['passed']:
                        console.print("[green]✅ Quantization passed![/green]")
                    else:
                        console.print("[yellow]⚠️ Quantization warning! mAP drop > 5%[/yellow]")
                return
                
            except ImportError:
                console.print("[yellow]⚠️[/] YOLO validation not available (install ultralytics)")
                console.print("   Falling back to basic validation...")
            except Exception as e:
                console.print(f"[yellow]⚠️[/] YOLO validation failed: {e}")
                console.print("   Falling back to basic validation...")
        else:
            if data_yaml:
                console.print(f"[yellow]⚠️[/] data.yaml not found: {data_yaml}")
            console.print("   Using basic validation (no mAP calculation)")
            console.print("   For mAP metrics, provide --data-yaml")
        
        # Fallback to basic validation
        validation_results = validate_quantization(
            float32_model_path=onnx_path,
            quantized_model_path=quantized_path,
            validation_folder=calib,
            input_name=input_name,
            input_shape=(height, width, channels),
            batch_size=batch_size,
            max_samples=max_samples,
            channel_order=channel_order
        )
        
        if validation_results:
            float32_acc = validation_results['float32'].get('accuracy_percent', 0)
            quantized_acc = validation_results['quantized'].get('accuracy_percent', 0)
            delta = float32_acc - quantized_acc if float32_acc and quantized_acc else 0
            
            if delta < 5:
                console.print(f"[green]✅ Validation passed! (Accuracy drop: {delta:.2f}%)[/green]")
            else:
                console.print(f"[yellow]⚠️ Validation warning: Accuracy drop of {delta:.2f}%[/yellow]")
                console.print("   Consider using more calibration samples or different quantization settings.")
        
    except ImportError as e:
        console.print(f"[yellow]⚠️[/] Validation module not available: {e}")
    except Exception as e:
        console.print(f"[yellow]⚠️[/] Validation failed: {e}")
        console.print("   Skipping validation...")


if __name__ == '__main__':
    main()