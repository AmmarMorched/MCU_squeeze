# mcusqueeze/quantization/ptq.py

"""
Post-Training Quantization (PTQ) module for ONNX models.

This module handles the quantization of float32 ONNX models to int8
using ONNX Runtime's quantization tools with calibration data.
"""

import os
import onnx
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationMethod,
    QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .calibration_reader import ONNXCalibrationDataReader
from .calibration import CalibrationDataset


class PTQ:
    """
    Post-Training Quantization for ONNX models.
    
    This class handles the complete quantization pipeline:
    1. Load float32 ONNX model
    2. Run calibration on image dataset
    3. Quantize weights and activations to int8
    4. Save quantized ONNX model
    
    Example:
        >>> quantizer = PTQ(
        ...     model_path='model.onnx',
        ...     output_path='model_quantized.onnx',
        ...     target='esp32s3'
        ... )
        >>> quantizer.quantize(
        ...     calibration_folder='images/',
        ...     input_name='input',
        ...     input_shape=(224, 224, 3)
        ... )
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        target: str = 'esp32s3',
        preprocess: bool = True,
    ):
        """
        Initialize the PTQ quantizer.
        
        Args:
            model_path: Path to input float32 ONNX model
            output_path: Path to output quantized ONNX model
            target: Target MCU (esp32s3, stm32, etc.)
            preprocess: Whether to run ONNX shape inference before quantization
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.target = target
        self.preprocess = preprocess
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_path.exists():
            self.original_size = self.model_path.stat().st_size/(1024*1024)
        else:
            self.original_size = 0

        self.quantized_size = 0
        self.quantization_params = {}
        self.layer_stats = {}
        
        # Validate input model exists
        if  self.model_path.exists():
            print(f"📦 Original model size: {self.original_size:.2f} MB")
        else:
            print(f"⚠️ Model not found: {self.model_path}")
        
        # Get model size
        #self.original_size = self.model_path.stat().st_size / (1024 * 1024)
        print(f"📦 Original model size: {self.original_size:.2f} MB")
    
    def quantize(
        self,
        calibration_folder: str,
        input_name: str,
        input_shape: Tuple[int, int, int],
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        per_channel: bool = True,
        activation_type: QuantType = QuantType.QInt8,
        weight_type: QuantType = QuantType.QInt8,
        calibration_method: CalibrationMethod = CalibrationMethod.MinMax,
        quant_format: QuantFormat = QuantFormat.QDQ,
        extra_options: Optional[Dict[str, Any]] = None,
        channel_order: str = 'NHWC',
    ) -> str:
        """
        Quantize the model using static quantization with calibration.
        
        Args:
            calibration_folder: Folder containing calibration images
            input_name: Name of the model's input node
            input_shape: Expected input shape (height, width, channels)
            batch_size: Batch size for calibration
            max_samples: Maximum number of samples to use (None = all)
            per_channel: Enable per-channel quantization
            activation_type: Quantization type for activations
            weight_type: Quantization type for weights
            calibration_method: Calibration method (MinMax, Entropy, Percentile)
            quant_format: Quantization format (QDQ or QOperator)
            extra_options: Additional quantization options
        
        Returns:
            Path to the quantized ONNX model
        """
        
        print(f"\n🔧 Starting Post-Training Quantization")
        print(f"   Model: {self.model_path}")
        print(f"   Target: {self.target}")
        print(f"   Calibration: {calibration_folder}")
        print(f"   Input: {input_name} {input_shape}")
        print(f"   Batch size: {batch_size}")
        print(f"   Per-channel: {per_channel}")
        print(f"   Calibration method: {calibration_method}")
        print("-" * 60)
        
        # 1. Validate calibration folder
        calib_path = Path(calibration_folder)
        if not calib_path.exists():
            raise ValueError(f"Calibration folder not found: {calibration_folder}")
        
        # 2. Preprocess model (optional)
        preprocessed_path = None
        if self.preprocess:
            print("🔧 Running shape inference and model preprocessing...")
            preprocessed_path = self._preprocess_model()
            model_path_for_quant = preprocessed_path
        else:
            model_path_for_quant = self.model_path
        
        # 3. Create calibration data reader
        print("📊 Creating calibration data reader...")
        cache_dir=str(Path(calibration_folder) / ".cache"),
        calibration_data_reader = ONNXCalibrationDataReader(
            folder_path=calibration_folder,
            input_name=input_name,
            input_shape=input_shape,
            batch_size=batch_size,
            max_samples=max_samples,
            normalize=True,
            channel_order=channel_order,
            cache_dir=cache_dir,
        )
        
        # 4. Set up quantization options
        print("🔧 Configuring quantization options...")
        
        # Default extra options
        if extra_options is None:
            extra_options = {}
        
        # Per-channel quantization options
        if per_channel:
            extra_options.update({
                'ActivationSymmetric': True,
                'WeightSymmetric': True,
                'EnableSubgraph': True,
            })
        
        # 5. Run quantization
        print("⚡ Running quantization...")
        print("   This may take a few minutes...")
        
        try:
            quantize_static(
                model_input=str(self.model_path),
                model_output=str(self.output_path),
                calibration_data_reader=calibration_data_reader,
                quant_format=quant_format,
                activation_type=activation_type,
                weight_type=weight_type,
                per_channel=per_channel,
                calibrate_method=calibration_method,
                extra_options=extra_options,
            )
            print(f"✅ Quantization complete!")
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            raise
        
        # 6. Clean up preprocessed model
        if (self.preprocess and preprocessed_path and preprocessed_path.exists() and preprocessed_path != self.model_path):
            try:
                preprocessed_path.unlink()
                print(f"🧹 Cleaned up temporary file: {preprocessed_path}")
            except:
                pass
        
        # 7. Get quantized model size
        self.quantized_size = self.output_path.stat().st_size / (1024 * 1024)  # MB
        
        # 8. Print summary
        self._print_summary()
        
        return str(self.output_path)
    
    def _preprocess_model(self) -> Path:
        """
        Run ONNX shape inference and model preprocessing.
        
        Returns:
            Path to the preprocessed model file
        """
        preprocessed_path = self.model_path.parent / f"{self.model_path.stem}_preprocessed.onnx"

        # quant_pre_process calls onnx.save_model(..., "sym_shape_infer_temp.onnx",
        # save_as_external_data=True) relative to the CURRENT working directory, which
        # leaves an orphaned "<uuid>.data" weight file (and sym_shape_infer_temp.onnx)
        # next to cwd on every run. Sandbox the call in a temp dir and clean it up.
        import tempfile

        try:
            with tempfile.TemporaryDirectory(prefix="mcusqueeze_preproc_") as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    quant_pre_process(
                        str(self.model_path),
                        str(preprocessed_path),
                    )
                finally:
                    os.chdir(cwd)
            return preprocessed_path
        except Exception as e:
            print(f"⚠️ Model preprocessing warning: {e}")
            print("   Using original model...")
            return self.model_path
    
    def _print_summary(self):
        """Print quantization summary."""
        reduction = (1 - self.quantized_size / self.original_size) * 100 if self.original_size > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 Quantization Results")
        print("=" * 60)
        print(f"   Original size:  {self.original_size:.2f} MB")
        print(f"   Quantized size: {self.quantized_size:.2f} MB")
        print(f"   Reduction:      {reduction:.1f}%")
        print(f"   Target:         {self.target}")
        print("=" * 60)
        print(f"   Output saved to: {self.output_path}")
        print("=" * 60)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model before/after quantization.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'original_size_mb': self.original_size,
            'quantized_size_mb': self.quantized_size,
            'reduction_percent': (1 - self.quantized_size / self.original_size) * 100 if self.original_size > 0 else 0,
            'target': self.target,
            'model_path': str(self.model_path),
            'output_path': str(self.output_path),
        }


def quantize_model(
    model_path: str,
    output_path: str,
    calibration_folder: str,
    input_name: str,
    input_shape: Tuple[int, int, int],
    target: str = 'esp32s3',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    per_channel: bool = True,
) -> str:
    """
    Convenience function to quantize an ONNX model.
    
    This is a simpler wrapper around the PTQ class.
    
    Args:
        model_path: Path to input float32 ONNX model
        output_path: Path to output quantized ONNX model
        calibration_folder: Folder containing calibration images
        input_name: Name of the model's input node
        input_shape: Expected input shape (height, width, channels)
        target: Target MCU
        batch_size: Batch size for calibration
        max_samples: Maximum samples to use
        per_channel: Enable per-channel quantization
    
    Returns:
        Path to the quantized ONNX model
    
    Example:
        >>> quantize_model(
        ...     model_path='model.onnx',
        ...     output_path='model_quantized.onnx',
        ...     calibration_folder='images/',
        ...     input_name='input',
        ...     input_shape=(224, 224, 3),
        ... )
    """
    quantizer = PTQ(
        model_path=model_path,
        output_path=output_path,
        target=target,
    )
    
    return quantizer.quantize(
        calibration_folder=calibration_folder,
        input_name=input_name,
        input_shape=input_shape,
        batch_size=batch_size,
        max_samples=max_samples,
        per_channel=per_channel,
    )


def get_quantization_options_for_target(target: str) -> Dict[str, Any]:
    """
    Get recommended quantization options for a specific target MCU.
    
    Args:
        target: Target MCU name
    
    Returns:
        Dictionary of recommended options
    """
    options = {
        'esp32s3': {
            'activation_type': QuantType.QInt8,
            'weight_type': QuantType.QInt8,
            'per_channel': True,
            'calibration_method': CalibrationMethod.MinMax,
            'quant_format': QuantFormat.QDQ,
            'extra_options': {
                'ActivationSymmetric': True,
                'WeightSymmetric': True,
                'EnableSubgraph': True,
            },
        },
        'stm32': {
            'activation_type': QuantType.QInt8,
            'weight_type': QuantType.QInt8,
            'per_channel': True,
            'calibration_method': CalibrationMethod.MinMax,
            'quant_format': QuantFormat.QDQ,
            'extra_options': {
                'ActivationSymmetric': True,
                'WeightSymmetric': True,
            },
        },
        'rp2040': {
            'activation_type': QuantType.QUInt8,
            'weight_type': QuantType.QUInt8,
            'per_channel': False,
            'calibration_method': CalibrationMethod.Entropy,
            'quant_format': QuantFormat.QOperator,
            'extra_options': {},
        },
    }
    
    return options.get(target, options['esp32s3'])


