from .esp32s3 import ESP32S3

SUPPORTED_TARGETS = {
    "esp32s3": ESP32S3,
}

DEFAULT_TARGET='esp32s3'


def get_target(target_name: str):
    if target_name not in SUPPORTED_TARGETS:
        return None
    return SUPPORTED_TARGETS[target_name]


def get_available_targets():
    return list(SUPPORTED_TARGETS.keys())

def get_target_names_for_help():
    return ", ".join(get_available_targets())