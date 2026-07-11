# mcusqueeze/quantization/calibration_reader.py

"""
ONNX Runtime Calibration Data Reader.

This module provides a bridge between our CalibrationDataset
and ONNX Runtime's quantization API.
"""

import numpy as np
from typing import Optional, Dict, Iterator
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader

from .calibration import CalibrationDataset


class ONNXCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for ONNX Runtime quantization.
    
    This class adapts our CalibrationDataset to the format expected
    by ONNX Runtime's quantize_static function.
    """
    
    def __init__(
        self,
        folder_path: str,
        input_name: str,
        input_shape: tuple,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        normalize: bool = True,
        channel_order: str = 'NHWC',
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the calibration data reader.
        
        Args:
            folder_path: Path to calibration images
            input_name: Name of the model's input node
            input_shape: Expected input shape (H, W, C)
            batch_size: Batch size for calibration
            max_samples: Maximum samples to use
            normalize: Whether to normalize images
            channel_order: Order of channels in the input (NHWC or NCHW)
            cache_dir: Directory to store temporary files (optional)
        """
        self.input_name = input_name
        self.channel_order = channel_order
        
        # ✅ Fix: Ensure cache_dir is a string, not a tuple
        if cache_dir is None:
            cache_dir = str(Path(folder_path) / ".cache")
        elif isinstance(cache_dir, tuple):
            # If it's a tuple, take the first element
            cache_dir = str(cache_dir[0]) if cache_dir else str(Path(folder_path) / ".cache")
        else:
            cache_dir = str(cache_dir)
        
        self.cache_dir = cache_dir
        
        # ✅ Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.dataset = CalibrationDataset(
            folder_path=folder_path,
            input_shape=input_shape,
            batch_size=batch_size,
            max_samples=max_samples,
            normalize=normalize,
            channel_order=channel_order,
            cache_dir=self.cache_dir,
        )
        self._iterator = None
        
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the next calibration batch.
        
        Returns:
            Dictionary mapping input_name → batch data, or None if done
        """
        if self._iterator is None:
            self._iterator = iter(self.dataset)
        
        try:
            batch = next(self._iterator)
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32)
            return {self.input_name: batch}
        except StopIteration:
            return None
    
    def __iter__(self):
        """Iterator support for for-loop usage."""
        self._iterator = iter(self.dataset)
        return self
    
    def __next__(self):
        batch = next(self._iterator)
        return {self.input_name: batch}