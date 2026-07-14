# mcusqueeze/validation/validator.py

"""
Model-agnostic quantization validation.

This module compares a float32 (reference) ONNX model against its int8
quantized counterpart and reports how much accuracy / output fidelity was
lost during quantization.

It is intentionally model-agnostic: it works for classifiers, detectors,
segmenters, or any ONNX model, with or without ground-truth labels.

Two kinds of metrics are produced:

1. Numerical fidelity (every model, every output tensor):
     - max absolute difference
     - mean absolute difference (L1)
     - mean relative error (%)
     - cosine similarity (direction preservation)
   These tell you *how far* the quantized outputs drifted from float32.

2. Prediction agreement (classifier-shaped outputs of shape [N, C]):
     - the fraction of samples where argmax(float32) == argmax(int8).
   This is the direct "did quantization flip the predicted class?" signal,
   i.e. the accuracy-loss proxy when no labels are available.

If ground-truth `labels` are supplied they are used to compute true
accuracy for both models as an extra check.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from mcusqueeze.quantization.calibration import get_calibration_data

console = Console()


class ModelValidator:
    """
    Validates quantized model accuracy/fidelity against a float32 model.
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
        labels: Optional[np.ndarray] = None,
    ):
        self.float32_path = Path(float32_model_path)
        self.quantized_path = Path(quantized_model_path)
        self.validation_folder = validation_folder
        self.input_shape = input_shape
        self.height, self.width, self.channels = input_shape
        self.input_name = input_name
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.channel_order = channel_order
        self.labels = labels

        # Results
        self.float32_stats: Dict[str, Any] = {}
        self.quantized_stats: Dict[str, Any] = {}
        self.comparison_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_session(path: str):
        import onnxruntime as ort
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    def _run_inference(self, session, name: str) -> Dict[str, np.ndarray]:
        """Run the model over the whole validation set, return {output_name: array}."""
        out_names = [o.name for o in session.get_outputs()]
        collected: Dict[str, List[np.ndarray]] = {n: [] for n in out_names}
        actual_input = session.get_inputs()[0].name

        console.print(f"\n   Running inference on {name} model...")
        n = 0
        for batch in get_calibration_data(
            folder_path=self.validation_folder,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
            channel_order=self.channel_order,
        ):
            n += 1
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32)
            try:
                outs = session.run(None, {actual_input: batch})
            except Exception as e:
                console.print(f"      [red]✗ Inference failed for batch {n}: {e}[/]")
                continue
            for o_name, arr in zip(out_names, outs):
                collected[o_name].append(np.asarray(arr))

        if not any(collected.values()):
            raise RuntimeError("No batches were successfully processed.")

        result = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
        total = sum(arr.shape[0] for arr in result.values()) // max(len(result), 1)
        console.print(f"   [green]✓[/] Processed {total} samples across {len(result)} output(s)")
        return result

    @staticmethod
    def _fidelity(f: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Per-output numerical fidelity between float32 and quantized."""
        f_flat = f.reshape(f.shape[0], -1).astype(np.float64)
        q_flat = q.reshape(q.shape[0], -1).astype(np.float64)
        diff = np.abs(q_flat - f_flat)

        max_abs = float(diff.max())
        mean_abs = float(diff.mean())

        # Mean relative error per sample, then averaged.
        denom = np.mean(np.abs(f_flat), axis=1) + 1e-9
        rel_err = float(np.mean(diff.mean(axis=1) / denom) * 100.0)

        # Cosine similarity per sample, averaged.
        fn = np.linalg.norm(f_flat, axis=1) + 1e-12
        qn = np.linalg.norm(q_flat, axis=1) + 1e-12
        cos = np.sum(f_flat * q_flat, axis=1) / (fn * qn)
        cos_mean = float(np.mean(cos))

        return {
            'max_abs': max_abs,
            'mean_abs': mean_abs,
            'rel_err_pct': rel_err,
            'cosine': cos_mean,
        }

    @staticmethod
    def _agreement(f: np.ndarray, q: np.ndarray) -> Optional[float]:
        """Class agreement for classifier-shaped [N, C] outputs."""
        if len(f.shape) != 2 or f.shape[0] == 0:
            return None
        fc = np.argmax(f.reshape(f.shape[0], -1), axis=1)
        qc = np.argmax(q.reshape(q.shape[0], -1), axis=1)
        return float(np.mean(fc == qc) * 100.0)

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def validate(self) -> Dict[str, Any]:
        console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print("[bold cyan]🔬 Quantization Validation[/bold cyan]")
        console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

        console.print(f"\n📁 Validation folder: {self.validation_folder}")
        console.print(f"📐 Input shape: {self.input_shape}")
        console.print(f"📦 Batch size: {self.batch_size}")

        float32_session = self._load_session(str(self.float32_path))
        quantized_session = self._load_session(str(self.quantized_path))

        f_preds = self._run_inference(float32_session, "float32")
        q_preds = self._run_inference(quantized_session, "quantized")

        self.float32_stats['model_size_mb'] = self.float32_path.stat().st_size / (1024 * 1024)
        self.quantized_stats['model_size_mb'] = self.quantized_path.stat().st_size / (1024 * 1024)

        # Compare outputs by position (robust to renamed/dequant outputs).
        f_list = list(f_preds.values())
        q_list = list(q_preds.values())
        n_out = min(len(f_list), len(q_list))

        per_output: List[Dict[str, Any]] = []
        agreements: List[float] = []
        for i in range(n_out):
            f_arr = f_list[i]
            q_arr = q_list[i]
            entry = {'index': i, 'shape': list(f_arr.shape)}
            if f_arr.shape != q_arr.shape:
                entry['shape_mismatch'] = True
                entry['quant_shape'] = list(q_arr.shape)
                per_output.append(entry)
                continue
            fid = self._fidelity(f_arr, q_arr)
            entry.update(fid)
            agr = self._agreement(f_arr, q_arr)
            if agr is not None:
                entry['agreement'] = agr
                agreements.append(agr)
            per_output.append(entry)

        self.comparison_stats = {
            'per_output': per_output,
            'n_outputs': n_out,
            'mean_agreement': float(np.mean(agreements)) if agreements else None,
        }

        # Optional true-accuracy (if labels provided).
        if self.labels is not None and n_out > 0 and len(f_list[0].shape) == 2:
            self._true_accuracy(f_list[0], q_list[0])

        self._print_results()
        return {
            'float32': self.float32_stats,
            'quantized': self.quantized_stats,
            'comparison': self.comparison_stats,
        }

    def _true_accuracy(self, f: np.ndarray, q: np.ndarray):
        labels = self.labels
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=1)
        f_pred = np.argmax(f.reshape(f.shape[0], -1), axis=1)
        q_pred = np.argmax(q.reshape(q.shape[0], -1), axis=1)
        fa = float(np.mean(f_pred == labels) * 100.0)
        qa = float(np.mean(q_pred == labels) * 100.0)
        self.float32_stats['accuracy_percent'] = fa
        self.quantized_stats['accuracy_percent'] = qa
        self.comparison_stats['true_accuracy_delta'] = fa - qa

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _print_results(self):
        console.print("\n[bold green]📊 Validation Results[/bold green]")
        console.print("-" * 60)

        # Model size
        console.print("\n[bold]📦 Model Size Comparison:[/bold]")
        console.print(f"   Float32:  {self.float32_stats['model_size_mb']:.2f} MB")
        console.print(f"   int8:     {self.quantized_stats['model_size_mb']:.2f} MB")
        size_red = (1 - self.quantized_stats['model_size_mb'] / self.float32_stats['model_size_mb']) * 100
        console.print(f"   [green]Reduction: {size_red:.1f}%[/green]")

        # Optional true accuracy
        if 'accuracy_percent' in self.float32_stats:
            fa = self.float32_stats['accuracy_percent']
            qa = self.quantized_stats['accuracy_percent']
            delta = fa - qa
            console.print("\n[bold]🎯 True Accuracy (labels provided):[/bold]")
            console.print(f"   Float32:  {fa:.2f}%")
            console.print(f"   int8:     {qa:.2f}%")
            console.print(f"   [yellow]Delta:    {delta:+.2f}%[/yellow]")

        # Per-output fidelity table
        console.print("\n[bold]🔍 Output Fidelity (float32 vs int8):[/bold]")
        table = Table(show_lines=False)
        table.add_column("Output")
        table.add_column("Cosine", justify="right")
        table.add_column("RelErr%", justify="right")
        table.add_column("MaxAbs", justify="right")
        table.add_column("MeanAbs", justify="right")
        table.add_column("Agree%", justify="right")
        for o in self.comparison_stats['per_output']:
            if o.get('shape_mismatch'):
                table.add_row(
                    f"#{o['index']}", "-", "-", "-", "-",
                    f"shape mismatch {o['shape']} vs {o.get('quant_shape')}",
                )
                continue
            agree = f"{o['agreement']:.1f}" if 'agreement' in o else "-"
            table.add_row(
                f"#{o['index']} {tuple(o['shape'])}",
                f"{o['cosine']:.4f}",
                f"{o['rel_err_pct']:.3f}",
                f"{o['max_abs']:.4f}",
                f"{o['mean_abs']:.4f}",
                agree,
            )
        console.print(table)

        # Prediction agreement (the accuracy-loss proxy)
        mean_agr = self.comparison_stats.get('mean_agreement')
        if mean_agr is not None:
            console.print(f"\n[bold]🎯 Prediction Agreement (float32→int8):[/bold] {mean_agr:.2f}%")
        else:
            console.print("\n[bold]🎯 Prediction Agreement:[/bold] n/a (no classifier-shaped [N,C] output)")

        # Verdict
        console.print("\n" + "-" * 60)
        self._verdict(mean_agr)

    def _verdict(self, mean_agr: Optional[float]):
        # Base verdict on fidelity of all outputs.
        rel_errs = [o['rel_err_pct'] for o in self.comparison_stats['per_output'] if 'rel_err_pct' in o]
        cosines = [o['cosine'] for o in self.comparison_stats['per_output'] if 'cosine' in o]
        mean_rel = float(np.mean(rel_errs)) if rel_errs else 100.0
        mean_cos = float(np.mean(cosines)) if cosines else 0.0

        if mean_agr is not None and mean_agr >= 99.0 and mean_rel < 1.0:
            console.print("[green]✅ Quantization preserved accuracy — negligible loss.[/green]")
        elif mean_agr is not None and mean_agr >= 95.0 and mean_rel < 5.0:
            console.print("[yellow]⚠️ Minor accuracy change — usually acceptable.[/yellow]")
        elif mean_cos >= 0.999 and mean_rel < 10.0:
            console.print("[yellow]⚠️ Outputs drifted slightly; no class-metric available — review.[/yellow]")
        else:
            console.print("[red]❌ Significant accuracy/quality loss detected.[/red]")
            console.print("   Consider: more calibration samples, per-channel quantization,")
            console.print("   or a different calibration method.")


def validate_quantization(
    float32_model_path: str,
    quantized_model_path: str,
    validation_folder: str,
    input_shape: Tuple[int, int, int],
    input_name: str = 'input',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    channel_order: str = 'NHWC',
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run model-agnostic quantization validation.
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
        labels=labels,
    )
    return validator.validate()
