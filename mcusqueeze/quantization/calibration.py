# mcusqueeze/quantization/calibration.py

import os
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Iterator, Tuple, Optional, List
from tqdm import tqdm
import onnxruntime as ort


class CalibrationDataset:
    """
    Calibration dataset loader for quantization.
    Load images from a folder, preprocess them, and yield batches for calibration.
    """
    def __init__(self, 
                 folder_path: str, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),  # ✅ Fixed: 3 values
                 batch_size: int = 1,
                 max_samples: Optional[int] = None,
                 normalize: bool = True,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        
        self.folder_path = Path(folder_path)
        self.input_shape = input_shape
        self.height, self.width, self.channels = input_shape  # ✅ Now works with 3 values
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

        # Load image paths
        self.image_paths = self._get_image_paths()  # ✅ Fixed: method name and attribute

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {folder_path}")
        print(f"📸 Found {len(self.image_paths)} images for calibration")

    def _get_image_paths(self) -> List[Path]:  # ✅ Fixed: method name
        """
        Get all image file paths from the folder.
        """
        image_extensions = {'.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(self.folder_path.rglob(f'*{ext}'))
            image_paths.extend(self.folder_path.rglob(f'*{ext.upper()}'))

        # ✅ Fixed: variable name
        if self.max_samples and len(image_paths) > self.max_samples:
            image_paths = image_paths[:self.max_samples]
        return sorted(image_paths)
    
    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Load and process a single image.
        """
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not open image {image_path}: {e}")
            # Return blank image as fallback
            return np.zeros(self.input_shape, dtype=np.float32)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize((self.width, self.height), Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize
        if self.normalize:
            img_array = img_array / 255.0  # Scale to [0, 1]
            img_array = (img_array - self.mean) / self.std  # Normalize
        return img_array
    
    def __len__(self) -> int:
        """Total number of images."""
        return len(self.image_paths)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over batches."""
        batch = []

        for image_path in tqdm(self.image_paths, desc="Loading images"):
            try:
                img_array = self._preprocess_image(image_path)
                batch.append(img_array)
                if len(batch) == self.batch_size:
                    batch_array = np.stack(batch, axis=0)  # ✅ Fixed: axis not asis
                    yield batch_array
                    batch = []
            except Exception as e:
                print(f"⚠️ Warning: Could not process image {image_path}: {e}")
                continue

        if batch:
            batch_array = np.stack(batch, axis=0)  # ✅ Fixed: axis not asis
            yield batch_array

    def get_batch(self) -> np.ndarray:
        """Get a single batch for testing."""
        try:
            return next(iter(self))
        except StopIteration:
            return np.array([])
        
    def get_sample(self) -> np.ndarray:
        """Get one sample for testing."""
        sample_path = self.image_paths[0]
        return self._preprocess_image(sample_path)[np.newaxis, ...]


# ✅ Moved outside class - now a module-level function
def get_calibration_data(
    folder_path: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Convenience function to get calibration data iterator.
    
    Usage:
        for batch in get_calibration_data('images/', input_shape=(224,224,3)):
            # Run model inference on batch
            outputs = session.run(None, {'input': batch})
    """
    dataset = CalibrationDataset(
        folder_path=folder_path,
        input_shape=input_shape,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    for batch in dataset:
        yield batch