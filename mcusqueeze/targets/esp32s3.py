ESP32S3 = {

    # ─── Hardware Specs ──────────────────────────────────────
    "name":         "ESP32-S3",
    "vendor":       "Espressif",
    "cpu":          "Xtensa LX7 dual-core",
    "mhz":          240,
    "ram_kb":       512,        # internal SRAM
    "flash_kb":     8192,       # default 8MB flash
    "psram_kb":     8192,       # optional PSRAM (if enabled)
    "has_fpu":      True,
    "has_vector":   True,       # vector extensions for AI acceleration

    # ─── Runtime ─────────────────────────────────────────────
    "runtime":      "esp-dl",

    # ─── Supported ONNX Ops ──────────────────────────────────
    # based on ESP-DL supported layer list
    "supported_ops": [
        "Conv",
        "ConvTranspose",
        "DepthwiseConv",
        "Relu",
        "Relu6",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Gemm",
        "MatMul",
        "Add",
        "Mul",
        "BatchNormalization",
        "Flatten",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
        "Concat",
        "Slice",
        "Pad",
        "Resize",
        "Clip",
    ],

    # ─── Quantization ────────────────────────────────────────
    "supported_dtypes": ["int8", "int16"],
    "preferred_dtype":  "int8",
}