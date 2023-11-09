import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import numpy as np
import IPython.display as ipd
from scipy.signal.windows import hamming
from scipy.io import wavfile
import librosa
import soundfile as sf

def read_wav(filename):
    """
        read wav file.
        Normalizes signal to values between -1 and 1.
    """
    fs, s = wavfile.read(filename)  # scipy reads int
    s = np.array(s)/float(max(abs(s)))
    return fs,s

def enframe(x, win_len, hop_len):
    """
        receives a 1D numpy array and divides it into frames.
        outputs a numpy matrix with the frames on the rows.
    """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int64(np.floor((len(x) - win_len) / float(hop_len)))
    x_framed = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        x_framed[i] = x[i * hop_len : i * hop_len + win_len]
    return x_framed


def deframe(x_framed, win_len, hop_len):
    """
        interpolates 1D data with framed alignments into persample values.
        This function helps as a visual aid and can also be used to change 
        frame-rate for features, e.g. energy, zero-crossing, etc.
    """
    n_frames = len(x_framed)
    n_samples = (n_frames-1)*hop_len + win_len
    x_samples = np.zeros((n_samples,1))
    for i in range(n_frames):
        x_samples[i*hop_len : i*hop_len + win_len] = x_framed[i].reshape(-1,1)
    return x_samples

def compute_nrg(xframes):
    """
        calculate per frame energy
    """
    n_frames = xframes.shape[1]
    hamm_window = hamming(n_frames)
    weighted_matrix = xframes * hamm_window
    return np.diagonal(np.dot(weighted_matrix,weighted_matrix.T))/float(n_frames)

def compute_log_nrg(xframes):
    """
        calculate per frame energy in log
    """
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
    """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

#eps : kernel bandwidth     
def k_cal(x, y, eps=0.5):
    """
        calculate k value
    """
    dist = x - y
    p = -1*np.dot(dist, dist.T)/(2*eps*eps)
    return np.exp(p)
    
def nrg_vad(xframes,thr=0.2):
    xframes = zero_mean(xframes) #u0..u[N]
    n_frames = xframes.shape[0]
    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes) #f0..f[N]
    xvad = np.zeros((n_frames,1))
    for i in range(1, n_frames):
        if k_cal(xframes[0], xframes[i]) <= thr:
            xvad[i] = 1.
        else:
            xvad[i] = 0.
    return xvad

def trim_and_concat_all(speech, vad, samplerate):
    """
        trim the silence from the beginning and end of the speech
        concatenate all the speech segments
    """
    win_len = int(samplerate*0.025)
    hop_len = int(samplerate*0.010)
    speech = speech.squeeze()
    vad = vad.squeeze()
    start = 0
    end = 0
    for i in range(len(vad)):
        if vad[i] == 1:
            start = i
            break
    for i in range(len(vad)-1,0,-1):
        if vad[i] == 1:
            end = i
            break
    return speech[start*hop_len:end*hop_len]

def vad_one_file(src_path, des_folder= "") :
    """NOTICE THAT: Move pwd to the desire folder before running vad

    """
    samplerate, data = read_wav(src_path)
    ipd.Audio(data, rate=samplerate)

    win_len = int(samplerate*0.025)
    hop_len = int(samplerate*0.010)
    init_silence_len = int(0.1*samplerate) # it is assumed that the first 100ms doesnt contain any speech
    sframes = enframe(data,win_len,hop_len) # rows: frame index, cols: each frame

    vad2 = nrg_vad(sframes)
    x = deframe(vad2,win_len,hop_len)
    x = x.squeeze()

    speech = trim_and_concat_all(data, vad2, samplerate)

    filename = src_path.split("/")[-1]
    # save 
    sf.write(filename , speech, samplerate)
    return speech, samplerate

def voice_active_detection(paths) :
    data = []
    for path in paths :
        speech, sr = vad_one_file(path)
        data.append(speech)
    
    return data


