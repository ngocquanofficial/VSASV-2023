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

from src.fusion.model.lcnn import LCNN


def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis"""
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def _calc_stft(path: str) -> np.ndarray:
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
    print(amp_db.shape)
    amp_db = amp_db[:800, :].astype("float32")
    print(amp_db.shape)
    return amp_db[..., np.newaxis]

def calc_stft(paths_list: list) -> torch.Tensor :
    """

    This function extracts spectrograms from raw audio data by using FFT.

    Args:
     paths_list(list): List contains path to wav files

    Returns:
     data: spectrograms that have 4 dimentions like (n_paths_list, height, width, 1)
    """

    data = []
    for path in tqdm(paths_list):

        # Calculate STFT
        stft_spec = _calc_stft(path)
        data.append(stft_spec)

    np_output = np.array(data)
    return torch.from_numpy(np_output).permute(0, 3, 1, 2).to(torch.float32)


def _calc_cqt(path: str) -> np.ndarray:
    """Calculating CQT spectrogram

    Args:
        path (str): Path to audio file.

    Returns:
        np.ndarray: A CQT spectrogram.
    """
    y, sr = librosa.load(path)
    y = _preEmphasis(y)
    cqt_spec = librosa.core.cqt(y, sr= sr)
    cq_db = librosa.amplitude_to_db(np.abs(cqt_spec))  # Amplitude to dB.
    return cq_db


def calc_cqt(paths_list: list, dir= "") -> torch.Tensor :
    """Calculate spectrograms from raw audio data by using CQT.

    Please refer to `calc_stft` for arguments and returns
    They are almost same.
    """
    max_width = 200  # for resizing cqt spectrogram.

    for i, path in enumerate(tqdm(paths_list)):
        full_path = dir + path
        # Calculate CQT spectrogram
        cqt_spec = _calc_cqt(full_path)

        height = cqt_spec.shape[0]
        if i == 0:
            resized_data = np.zeros((len(paths_list), height, max_width))

        # Truncate
        if max_width <= cqt_spec.shape[1]:
            cqt_spec = cqt_spec[:, :max_width]
        else:
            # Zero padding
            diff = max_width - cqt_spec.shape[1]
            zeros = np.zeros((height, diff))
            cqt_spec = np.concatenate([cqt_spec, zeros], 1)

        resized_data[i] = np.float32(cqt_spec)

    # # Extract labels from protocol
    # labels = _extract_label(protocol_df)

    np_output = resized_data[..., np.newaxis]
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
        feature (np.ndarray): Spectrograms with 4 dimensional shape like (n_paths_list, height, width, 1)
        path (str): Path for saving.
    """
    with open(path, "wb") as web:
        pickle.dump(feature, web, protocol=4)
        
# TEST
filenames = ["src/fusion/file_example_WAV_2MG.wav", "src/fusion/file_example_WAV_1MG.wav"]
data = calc_stft(filenames)
# data = torch.from_numpy(data).permute(0, 3, 1, 2)

data_cqt = calc_cqt(filenames)
# data_cqt = torch.from_numpy(data_cqt).permute(0, 3, 1, 2)
# data_cqt = data_cqt.to(torch.float32)
print(data_cqt)

model = LCNN(input_dim= 1, num_label= 50)
# print(model(data).shape)

print(model(data_cqt).shape)