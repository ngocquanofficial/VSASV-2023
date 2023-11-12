import os
import sys 
sys.path.append(os.getcwd()) # NOQA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

import pickle
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from src.fusion.LCNN.model.lcnn import LCNN


def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis"""
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def calc_stft_one_file(wave, sr) :
    """Calculate STFT with librosa.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: A STFT spectrogram.
    """
    # wave, sr = librosa.load(path)
    wave = _preEmphasis(wave)
    steps = int(len(wave) * 0.0081)
    # calculate STFT
    stft = librosa.stft(wave, n_fft=sr, win_length=1700, hop_length=steps, window="blackman")
    amp_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    amp_db = amp_db[:800, :].astype("float32")
    np_output = amp_db[..., np.newaxis]
    
    return torch.from_numpy(np_output).to(torch.float32)

def calc_cqt_one_file(wave, sr) -> np.ndarray:
    """Calculating CQT spectrogram

    Args:
        path (str): Path to audio file.

    Returns:
        np.ndarray: A CQT spectrogram.
    """
    # y, sr = librosa.load(path)
    y = _preEmphasis(wave)
    cqt_spec = librosa.core.cqt(y, sr=sr)
    cq_db = librosa.amplitude_to_db(np.abs(cqt_spec))  # Amplitude to dB.

    cqt_spec = cq_db
    height = cqt_spec.shape[0]
    max_width = 200  # for resizing cqt spectrogram.

    # Truncate
    if max_width <= cqt_spec.shape[1]:
        cqt_spec = cqt_spec[:, :max_width]
    else:
        # Zero padding
        diff = max_width - cqt_spec.shape[1]
        zeros = np.zeros((height, diff))
        cqt_spec = np.concatenate([cqt_spec, zeros], 1)
    
    np_output = cqt_spec[..., np.newaxis]
    
    return torch.from_numpy(np_output).to(torch.float32)
import numpy as np
import scipy.signal
import librosa
import torch

def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis"""
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def calc_mel_one_file(wave, sr):
    """Calculate Mel spectrogram with librosa.

    Args:
        wave (np.ndarray): Audio waveform.
        sr (int): Sampling rate.

    Returns:
        torch.Tensor: A Mel spectrogram.
    """
    # Apply pre-emphasis
    wave = _preEmphasis(wave)
    steps = int(len(wave) * 0.0081)

    # Calculate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=wave, n_fft=sr, win_length=1700, hop_length=steps, window="blackman",  n_mels=128)

    # Convert to decibels
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max).astype("float32")

    # Crop or pad to a fixed size if needed
    mel_db = mel_db[:, :800]

    # Add channel dimension
    mel_db = mel_db[..., np.newaxis]

    # Convert to PyTorch tensor
    torch_output = torch.from_numpy(mel_db).to(torch.float32)

    return torch_output
