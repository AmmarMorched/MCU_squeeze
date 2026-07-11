# mcusqueeze/quantization/calibration.py

from pathlib import Path
import numpy as np
from PIL import Image
from typing import Iterator, Tuple, Optional, List
from tqdm import tqdm


class CalibrationDataset:
    """
    Calibration dataset loader for quantization.
    Load images from a folder, preprocess them, and yield batches for calibration.
    """
    def __init__(self, 
                 folder_path: str, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 batch_size: int = 1,
                 max_samples: Optional[int] = None,
                 normalize: bool = True,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 channel_order: str = 'NHWC',
                 cache_dir: Optional[str] = None,
                 ):
        
        self.folder_path = Path(folder_path)
        self.input_shape = input_shape
        self.height, self.width, self.channels = input_shape
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        self.channel_order = channel_order
        
        # ✅ Fix: Ensure cache_dir is a string
        if cache_dir is None:
            cache_dir = str(self.folder_path / ".cache")
        elif isinstance(cache_dir, tuple):
            cache_dir = str(cache_dir[0]) if cache_dir else str(self.folder_path / ".cache")
        else:
            cache_dir = str(cache_dir)
        
        self.cache_dir = cache_dir
        
        # ✅ Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.original_cwd = None

        # Load image paths
        self.image_paths = self._get_image_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {folder_path}")
        print(f"📸 Found {len(self.image_paths)} images for calibration")

    def _get_image_paths(self) -> List[Path]:
        """
        Get all image file paths from the folder.
        """
        image_extensions = {'.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_paths = []

        for ext in image_extensions:
            for p in self.folder_path.rglob(f'*{ext}'):
                image_paths.append(p.resolve())
            for p in self.folder_path.rglob(f'*{ext.upper()}'):
                image_paths.append(p.resolve())

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
            img_array = img_array / 255.0
            img_array = (img_array - self.mean) / self.std
        
        # Channel order conversion
        if self.channel_order == 'NCHW':
            img_array = np.transpose(img_array, (2, 0, 1))
        
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
                    batch_array = np.stack(batch, axis=0)
                    if batch_array.dtype != np.float32:
                        batch_array = batch_array.astype(np.float32)
                    yield batch_array
                    batch = []
            except Exception as e:
                print(f"⚠️ Warning: Could not process image {image_path}: {e}")
                continue

            if batch:
                batch_array = np.stack(batch, axis=0)
                if batch_array.dtype != np.float32:
                    batch_array = batch_array.astype(np.float32)
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


def get_calibration_data(
    folder_path: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    channel_order: str = 'NHWC',
    cache_dir: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """
    Convenience function to get calibration data iterator.
    
    Args:
        folder_path: Path to calibration images
        input_shape: Expected input shape (H, W, C)
        batch_size: Batch size for calibration
        max_samples: Maximum samples to use
        channel_order: Order of channels (NHWC or NCHW)
        cache_dir: Directory to store temporary files
    """
    dataset = CalibrationDataset(
        folder_path=folder_path,
        input_shape=input_shape,
        batch_size=batch_size,
        max_samples=max_samples,
        channel_order=channel_order,
        cache_dir=cache_dir,
    )
    
    for batch in dataset:
        yield batch