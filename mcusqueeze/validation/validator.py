# mcusqueeze/validation/validator.py

"""
Quantization Validation Module
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mcusqueeze.quantization.calibration import get_calibration_data

console = Console()


class ModelValidator:
    """
    Validates quantized model accuracy against float32 model.
    """
    
    def __init__(
        self,
        float32_model_path: str,
        quantized_model_path: str,
        validation_folder: str,
        input_shape: Tuple[int, int, int],
        input_name: str = 'input',
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        channel_order: str = 'NHWC',
    ):
        """
        Initialize the validator.
        """
        self.float32_path = Path(float32_model_path)
        self.quantized_path = Path(quantized_model_path)
        self.validation_folder = validation_folder
        self.input_shape = input_shape
        self.height, self.width, self.channels = input_shape
        self.input_name = input_name
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.channel_order = channel_order
        
        # Results storage
        self.float32_predictions = []
        self.quantized_predictions = []
        self.labels = []
        self.images = []
        
        # Statistics
        self.float32_stats = {}
        self.quantized_stats = {}
        self.comparison_stats = {}
    
    def _load_models(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """Load both float32 and quantized models."""
        console.print("[cyan]→[/] Loading models...")
        
        if not self.float32_path.exists():
            raise FileNotFoundError(f"Float32 model not found: {self.float32_path}")
        
        if not self.quantized_path.exists():
            raise FileNotFoundError(f"Quantized model not found: {self.quantized_path}")
        
        console.print(f"   Loading float32 model: {self.float32_path.name}")
        float32_session = ort.InferenceSession(
            str(self.float32_path),
            providers=['CPUExecutionProvider']
        )
        
        console.print(f"   Loading quantized model: {self.quantized_path.name}")
        quantized_session = ort.InferenceSession(
            str(self.quantized_path),
            providers=['CPUExecutionProvider']
        )
        
        return float32_session, quantized_session
    
    def _get_predictions(
        self,
        session: ort.InferenceSession,
        name: str
    ) -> List[np.ndarray]:
        """Run inference on all validation images."""
        predictions = []
        batch_count = 0
        total_samples = 0
        
        console.print(f"\n   Running inference on {name} model...")
        
        for batch in get_calibration_data(
            folder_path=self.validation_folder,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
            channel_order=self.channel_order,
        ):
            batch_count += 1
            total_samples += len(batch)
            
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32)
            
            try:
                outputs = session.run(None, {self.input_name: batch})
                predictions.extend(outputs[0])
            except Exception as e:
                console.print(f"      [red]✗ Inference failed for batch {batch_count}: {e}[/]")
                continue
        
        console.print(f"   [green]✓[/] Processed {len(predictions)} predictions")
        return np.array(predictions)
    
    def _calculate_accuracy(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate accuracy metrics."""
        metrics = {
            'total_samples': len(predictions),
            'prediction_shape': predictions.shape,
        }
        
        if len(predictions.shape) == 2:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        if labels is not None:
            if len(labels.shape) == 2:
                labels = np.argmax(labels, axis=1)
            
            correct = np.sum(pred_classes == labels)
            accuracy = correct / len(labels)
            
            metrics.update({
                'accuracy': accuracy,
                'accuracy_percent': accuracy * 100,
                'correct': correct,
                'total': len(labels),
            })
        
        if len(predictions.shape) == 2:
            confidences = np.max(predictions, axis=1)
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['min_confidence'] = np.min(confidences)
            metrics['max_confidence'] = np.max(confidences)
        
        return metrics
    
    def _compare_outputs(
        self,
        float32_preds: np.ndarray,
        quantized_preds: np.ndarray
    ) -> Dict[str, Any]:
        """Compare predictions from both models."""
        comparison = {}
        
        if float32_preds.shape != quantized_preds.shape:
            comparison['shape_mismatch'] = True
            comparison['float32_shape'] = float32_preds.shape
            comparison['quantized_shape'] = quantized_preds.shape
            return comparison
        
        comparison['shape_mismatch'] = False
        
        are_identical = np.allclose(float32_preds, quantized_preds, rtol=1e-3)
        comparison['identical'] = bool(are_identical)
        
        diff = float32_preds - quantized_preds
        comparison['max_diff'] = np.max(np.abs(diff))
        comparison['mean_diff'] = np.mean(np.abs(diff))
        comparison['std_diff'] = np.std(diff)
        
        if len(float32_preds.shape) == 2:
            float32_classes = np.argmax(float32_preds, axis=1)
            quantized_classes = np.argmax(quantized_preds, axis=1)
            
            agreement = np.sum(float32_classes == quantized_classes)
            comparison['class_agreement'] = agreement
            comparison['class_agreement_percent'] = (agreement / len(float32_classes)) * 100
        
        return comparison
    
    def validate(self) -> Dict[str, Any]:
        """Run the full validation pipeline."""
        # ✅ FIXED: Rich tags on same line
        console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print("[bold cyan]🔬 Quantization Validation[/bold cyan]")
        console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
        
        console.print(f"\n📁 Validation folder: {self.validation_folder}")
        console.print(f"📐 Input shape: {self.input_shape}")
        console.print(f"📦 Batch size: {self.batch_size}")
        
        # 1. Load models
        float32_session, quantized_session = self._load_models()
        
        # 2. Get predictions from float32 model
        console.print("\n[cyan]→[/] Running inference on float32 model...")
        float32_preds = self._get_predictions(float32_session, "float32")
        
        # 3. Get predictions from quantized model
        console.print("\n[cyan]→[/] Running inference on quantized model...")
        quantized_preds = self._get_predictions(quantized_session, "quantized")
        
        # 4. Calculate metrics
        console.print("\n[cyan]→[/] Calculating metrics...")
        
        self.float32_stats = self._calculate_accuracy(float32_preds)
        self.float32_stats['model_size_mb'] = self.float32_path.stat().st_size / (1024 * 1024)
        
        self.quantized_stats = self._calculate_accuracy(quantized_preds)
        self.quantized_stats['model_size_mb'] = self.quantized_path.stat().st_size / (1024 * 1024)
        
        self.comparison_stats = self._compare_outputs(float32_preds, quantized_preds)
        
        # 5. Display results
        self._print_results()
        
        return {
            'float32': self.float32_stats,
            'quantized': self.quantized_stats,
            'comparison': self.comparison_stats,
        }
    
    def _print_results(self):
        """Display validation results."""
        console.print("\n[bold green]📊 Validation Results[/bold green]")
        console.print("-" * 60)
        
        # Model Size Comparison
        console.print("\n[bold]📦 Model Size Comparison:[/bold]")
        console.print(f"   Float32:  {self.float32_stats['model_size_mb']:.2f} MB")
        console.print(f"   int8:     {self.quantized_stats['model_size_mb']:.2f} MB")
        
        size_reduction = (1 - self.quantized_stats['model_size_mb'] / self.float32_stats['model_size_mb']) * 100
        console.print(f"   [green]Reduction: {size_reduction:.1f}%[/green]")
        
        # Accuracy Comparison
        if 'accuracy' in self.float32_stats and 'accuracy' in self.quantized_stats:
            console.print("\n[bold]🎯 Accuracy Comparison:[/bold]")
            console.print(f"   Float32:  {self.float32_stats['accuracy_percent']:.2f}%")
            console.print(f"   int8:     {self.quantized_stats['accuracy_percent']:.2f}%")
            
            accuracy_delta = self.float32_stats['accuracy'] - self.quantized_stats['accuracy']
            console.print(f"   [yellow]Delta:    {accuracy_delta * 100:.2f}%[/yellow]")
            
            if accuracy_delta < 0.05:
                console.print("   [green]✅ Quantization passed! Accuracy drop is acceptable.[/green]")
            elif accuracy_delta < 0.10:
                console.print("   [yellow]⚠️ Quantization degraded accuracy. Consider reviewing.[/yellow]")
            else:
                console.print("   [red]❌ Quantization failed! Accuracy drop is too large.[/red]")
        
        # Comparison Stats
        console.print("\n[bold]🔍 Prediction Comparison:[/bold]")
        if self.comparison_stats.get('identical', False):
            console.print("   [green]✅ Predictions are identical![/green]")
        else:
            console.print(f"   Max difference: {self.comparison_stats.get('max_diff', 0):.6f}")
            console.print(f"   Mean difference: {self.comparison_stats.get('mean_diff', 0):.6f}")
            
            if 'class_agreement_percent' in self.comparison_stats:
                console.print(f"   Class agreement: {self.comparison_stats['class_agreement_percent']:.2f}%")
        
        console.print("\n" + "-" * 60)
        if 'accuracy' in self.float32_stats and 'accuracy' in self.quantized_stats:
            if self.float32_stats['accuracy'] - self.quantized_stats['accuracy'] < 0.05:
                console.print("[green]✅ Quantization validation PASSED[/green]")
            else:
                console.print("[yellow]⚠️ Quantization validation WARNING - Accuracy drop detected[/yellow]")
        else:
            console.print("[yellow]ℹ️ No labels provided - accuracy calculation skipped[/yellow]")


def validate_quantization(
    float32_model_path: str,
    quantized_model_path: str,
    validation_folder: str,
    input_shape: Tuple[int, int, int],
    input_name: str = 'input',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    channel_order: str = 'NHWC',
) -> Dict[str, Any]:
    """
    Convenience function to run quantization validation.
    """
    validator = ModelValidator(
        float32_model_path=float32_model_path,
        quantized_model_path=quantized_model_path,
        validation_folder=validation_folder,
        input_shape=input_shape,
        input_name=input_name,
        batch_size=batch_size,
        max_samples=max_samples,
        channel_order=channel_order,
    )
    
    return validator.validate()