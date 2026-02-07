#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions
"""

import platform
from pathlib import Path
import torch
from pysubs2.formats import FILE_EXTENSION_TO_FORMAT_IDENTIFIER


def _load_config(config_name, model_config, config_schema):
    """
    Helper function to load default values if `config_name` is not specified

    :param config_name: the name of the config
    :param model_config: configuration provided to the model
    :param config_schema: the schema of the configuration

    :return: config value
    """
    if config_name in model_config:
        return model_config[config_name]
    return config_schema[config_name]['default']


def get_available_devices() -> list:
    """
    Get available devices (cpu and gpus)

    :return: list of available devices
    """
    devices = ['cpu']
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    return devices


def detect_hardware() -> dict:
    """
    Detects runtime hardware capabilities relevant to model selection.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    has_mps = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    has_cuda = bool(torch.cuda.is_available())
    intel_gpu_detected = False

    if system == 'linux':
        intel_gpu_detected = _detect_linux_intel_gpu()

    return {
        'system': system,
        'machine': machine,
        'has_mps': has_mps,
        'has_cuda': has_cuda,
        'intel_gpu_detected': intel_gpu_detected,
    }


def select_faster_whisper_runtime(config_device: str, config_compute_type: str) -> tuple:
    """
    Resolve device/compute_type for faster-whisper using platform-aware defaults.
    Returns: (device, compute_type, reason)
    """
    hw = detect_hardware()
    reason = []
    device = config_device
    compute_type = config_compute_type

    if device == 'auto':
        if hw['system'] == 'linux' and hw['has_cuda']:
            device = 'cuda'
            reason.append('linux+nvidia-cuda detected')
            if compute_type == 'default':
                compute_type = 'float16'
        elif hw['system'] == 'darwin' and hw['machine'] in ('arm64', 'aarch64'):
            # faster-whisper uses CTranslate2 and does not expose an MPS backend.
            device = 'cpu'
            reason.append('apple-silicon detected')
            if compute_type == 'default':
                compute_type = 'int8'
        elif hw['system'] == 'linux' and hw['intel_gpu_detected']:
            raise RuntimeError(
                'Linux Intel GPU detected, but faster-whisper has no configured Intel GPU runtime in this project. '
                'Hard fail by policy.'
            )
        else:
            raise RuntimeError(
                f'Auto device selection failed for platform={hw["system"]} machine={hw["machine"]}. '
                'Hard fail by policy.'
            )
    elif device == 'cuda' and not hw['has_cuda']:
        raise RuntimeError('CUDA device was explicitly requested, but CUDA is unavailable on this system.')

    if not reason:
        reason.append('using explicit runtime configuration')

    return device, compute_type, '; '.join(reason)


def _detect_linux_intel_gpu() -> bool:
    """
    Best-effort Intel GPU detection from Linux sysfs.
    Intel vendor id is 0x8086.
    """
    drm_path = Path('/sys/class/drm')
    if not drm_path.exists():
        return False

    for vendor_file in drm_path.glob('card*/device/vendor'):
        try:
            if vendor_file.read_text(encoding='utf-8').strip().lower() == '0x8086':
                return True
        except OSError:
            continue

    return False


def available_translation_models() -> list:
    """
    Returns available translation models
    from (dl-translate)[https://github.com/xhluca/dl-translate]

    :return: list of available models
    """
    models = [
        "facebook/m2m100_418M",
        "facebook/m2m100_1.2B",
        "facebook/mbart-large-50-many-to-many-mmt",
        "facebook/nllb-200-distilled-600M"
    ]
    return models


def available_subs_formats(include_extensions=True):
    """
    Returns available subtitle formats
    from :attr:`pysubs2.FILE_EXTENSION_TO_FORMAT_IDENTIFIER`

    :param include_extensions: include the dot separator in file extensions

    :return: list of subtitle formats
    """

    extensions = list(FILE_EXTENSION_TO_FORMAT_IDENTIFIER.keys())

    if include_extensions:
        return extensions
    else:
        # remove the '.' separator from extension names
        return [ext.split('.')[1] for ext in extensions]
