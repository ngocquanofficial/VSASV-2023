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

def calc_stft_one_file(path) :
    """Calculate STFT with librosa.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: A STFT spectrogram.
    """
    wave, sr = librosa.load(path)
    wave = _preEmphasis(wave)
    steps = int(len(wave) * 0.0081)
    # calculate STFT
    stft = librosa.stft(wave, n_fft=sr, win_length=1700, hop_length=steps, window="blackman")
    amp_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    amp_db = amp_db[:800, :].astype("float32")
    np_output = amp_db[..., np.newaxis]
    
    return torch.from_numpy(np_output).to(torch.float32), sr

wave, sr= calc_stft_one_file("00000000.wav")
print(sr)