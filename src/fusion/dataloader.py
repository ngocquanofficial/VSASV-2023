import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import random
import torch
from torch.utils.data import Dataset

from src.fusion.LCNN.vad.vad import voice_active_detection, vad_one_file
from src.fusion.feature import calc_cqt_one_file, calc_stft_one_file

filenames = ["src/fusion/LCNN/vad/00000001.wav", "src/fusion/LCNN/vad/speech.wav"]
file = "src/fusion/LCNN/vad/00000001.wav"
speech, sr = vad_one_file(file)
data = voice_active_detection(filenames)

# use the embedding vector implemented from 2 model AASIST(antispoof-model) and ECAPA(verification-model)
class TrainingDataLCNN(Dataset) :
    def __init__(self, path_list,stft_embedding, cqt_embedding, mel_embedding, type= "stft") :
        self.type = type
        self.path_list = path_list
        self.stft_embedding = stft_embedding
        self.cqt_embedding = cqt_embedding
        self.mel_embedding = mel_embedding
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) :
        return len(self.stft_embedding)

    def __getitem__(self, index) :
        audio_path = random.choice(self.path_list)
        audio_type = audio_path.split("/")[-2]
        if audio_type == "bonafide" :
            label = 1
        elif audio_type == "spoofed_replay" or audio_type == "spoofed_voice_clone" :
            label = 0
        else :
            print("ERROR Occur at Dataloader")
            label = None
            print(audio_type, "LABEL")
        
        if self.type == "stft" :
            data = self.stft_embedding[audio_path]

        elif self.type == "cqt" :
            data = self.cqt_embedding[audio_path]
        elif self.type == "mel" :
            data = self.mel_embedding[audio_path]

        label = torch.tensor(label, dtype= torch.int64, device= self.device)
        return data.to(self.device), label
            
class ValidationDataLCNN(Dataset) :
    def __init__(self, path_list,stft_embedding, cqt_embedding, mel_embedding, type= "stft") :
        self.type = type
        self.path_list = path_list
        self.stft_embedding = stft_embedding
        self.cqt_embedding = cqt_embedding
        self.mel_embedding = mel_embedding
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) :
        return len(self.stft_embedding)

    def __getitem__(self, index) :
        audio_path = random.choice(self.path_list)
        audio_type = audio_path.split("/")[-2]
        if audio_type == "bonafide" :
            label = 1
        elif audio_type == "spoofed_replay" or audio_type == "spoofed_voice_clone" :
            label = 0
        else :
            print("ERROR Occur at Dataloader")
            label = None
            print(audio_type, "LABEL")
        
        if self.type == "stft" :
            data = self.stft_embedding[audio_path]
        elif self.type == "cqt" :
            data = self.cqt_embedding[audio_path]
        elif self.type == "mel" :
            data = self.mel_embedding[audio_path]

        label = torch.tensor(label, dtype= torch.int64, device= self.device)
        # print(data.shape, label)
        return data.to(self.device), label

class GenEmbDataLCNN:
    """
    Modify ValidationDataLCNN to work with generate embedding script
    """
    def __init__(self, path_list,stft_embedding, cqt_embedding, mel_embedding, type= "stft") :
        self.type = type
        self.path_list = list(set(path_list)) # note: path_list can contain duplicate values
        self.stft_embedding = stft_embedding
        self.cqt_embedding = cqt_embedding
        self.mel_embedding = mel_embedding
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) :
        return len(self.stft_embedding)

    def __getitem__(self, idx) :
        audio_path = self.path_list[idx]
        audio_type = audio_path.split("/")[-2]
        if audio_type == "bonafide" :
            label = 1
        elif audio_type == "spoofed_replay" or audio_type == "spoofed_voice_clone" :
            label = 0
        else :
            print("ERROR Occur at Dataloader")
            label = None
            print(audio_type, "LABEL")
        
        if self.type == "stft" :
            data = self.stft_embedding[audio_path]

        elif self.type == "cqt" :
            data = self.cqt_embedding[audio_path]
        
        elif self.type == "mel" :
            data = self.mel_embedding[audio_path]

        label = torch.tensor(label, dtype= torch.int64, device= self.device)
        # print(data.shape, label)
        return data.to(self.device), label, audio_path
