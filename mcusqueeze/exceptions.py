# mcusqueeze/exceptions.py


class MCUSqeezeError(Exception):
    """Base exception for all mcusqueeze errors."""
    pass


# ─── File Errors ────────────────────────────────────────────

class FileNotFoundError(MCUSqeezeError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"File not found: '{path}'\n"
            f"  Check the path and try again"
        )


class EmptyFileError(MCUSqeezeError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"File is empty: '{path}'"
        )


class DirectoryGivenError(MCUSqeezeError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"Expected a file, got a directory: '{path}'\n"
            f"  Provide a path to a .onnx or .h5 file"
        )


class PermissionError(MCUSqeezeError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"Cannot read file — permission denied: '{path}'"
        )


# ─── Format Errors ──────────────────────────────────────────

class UnsupportedFormatError(MCUSqeezeError):
    def __init__(self, ext, filename):
        self.ext = ext
        self.filename = filename
        super().__init__(
            f"Unsupported format: '{ext}'\n"
            f"  Supported formats: .onnx  .h5  .keras, .pt \n"
            f"  Got: {filename}"
        )


# ─── Conversion Errors ──────────────────────────────────────

class ConversionError(MCUSqeezeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(
            f"Model conversion failed:\n"
            f"  {reason}"
        )


class CustomLayerError(MCUSqeezeError):
    def __init__(self, layer_names: list):
        self.layer_names = layer_names
        super().__init__(
            f"Model contains custom layers that cannot be converted automatically:\n"
            f"  Layers: {', '.join(layer_names)}\n"
            f"  Rewrite them using standard ops before uploading"
        )


# ─── Validation Errors ──────────────────────────────────────

class InvalidONNXError(MCUSqeezeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(
            f"Invalid ONNX model:\n"
            f"  {reason}"
        )


class OpsetTooOldError(MCUSqeezeError):
    def __init__(self, opset, minimum=11):
        self.opset = opset
        self.minimum = minimum
        super().__init__(
            f"Opset {opset} too old — minimum opset {minimum} required\n"
            f"  Re-export your model with opset >= {minimum}"
        )


class DynamicShapeError(MCUSqeezeError):
    def __init__(self, input_name):
        self.input_name = input_name
        super().__init__(
            f"Dynamic shape detected in input '{input_name}'\n"
            f"  MCU deployment requires static input shapes\n"
            f"  Re-export with fixed input dimensions"
        )


class UnsupportedOpsError(MCUSqeezeError):
    def __init__(self, ops: list, target: str):
        self.ops = ops
        self.target = target
        super().__init__(
            f"Model contains ops not supported on {target}:\n"
            f"  Unsupported: {', '.join(ops)}\n"
            f"  Restructure the model to avoid these ops"
        )


# ─── Feasibility Errors ─────────────────────────────────────

class FlashExceededError(MCUSqeezeError):
    def __init__(self, model_kb, flash_kb, target):
        super().__init__(
            f"Model too large for {target} flash:\n"
            f"  Model size: {model_kb:.1f} KB\n"
            f"  Available:  {flash_kb} KB\n"
            f"  Reduce model size or target a larger MCU"
        )


class RAMExceededError(MCUSqeezeError):
    def __init__(self, required_kb, available_kb, target):
        super().__init__(
            f"Insufficient RAM on {target}:\n"
            f"  Required:  {required_kb:.1f} KB\n"
            f"  Available: {available_kb} KB\n"
            f"  Reduce model complexity or target a larger MCU"
        )