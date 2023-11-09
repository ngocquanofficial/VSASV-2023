import os
import sys 
sys.path.append(os.getcwd()) # NOQA
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
    y = _preEmphasis(y)
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
    
    return torch.from_numpy(np_output).permute(0, 3, 1, 2).to(torch.float32)


def _extract_label(protocol: pd.DataFrame) -> np.ndarray:
    """Extract labels from ASVSpoof2019 protocol

    Args:
        protocol (pd.DataFrame): ASVSpoof2019 protocol

    Returns:
        np.ndarray: Labels.
    """
    labels = np.ones(len(protocol))
    labels[protocol["key"] == "bonafide"] = 0
    return labels.astype(int)


def save_feature(feature: np.ndarray, path: str):
    """Save spectrograms as a binary file.

    Args:
        feature (np.ndarray): Spectrograms with 4 dimensional shape like (n_samples, height, width, 1)
        path (str): Path for saving.
    """
    with open(path, "wb") as web:
        pickle.dump(feature, web, protocol=4)