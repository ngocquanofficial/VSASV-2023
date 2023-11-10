import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import random
import torch
from torch.utils.data import Dataset
# use the embedding vector implemented from 2 model AASIST(antispoof-model) and ECAPA(verification-model)
class VietnamCeleb(Dataset) :
    def __init__(self, verification_embeddings, speaker_data) :

        self.verify_emb = verification_embeddings
        self.speaker_data = speaker_data # a dictionary, each key is the speaker id, value is a list contain the path to the wav file.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) :
        return len(self.verify_emb.keys())
    
    def __getitem__(self, index) :
        # Randomly create a label first, and then create data base on label_type

        label_type = random.choice([1, 0])
        if label_type == 1 : # which means 2 data files are from 1 person
            
            # Ensure that the speaker has at least 2 files
            speaker = random.choice(list(self.speaker_data.keys())) # random choice a speaker id
            target, second = random.sample(self.speaker_data[speaker], 2)
            
        elif label_type == 0 :
            first_speaker, second_speaker = random.sample(list(self.speaker_data.keys()), 2)
            target = random.choice(self.speaker_data[first_speaker])
            second = random.choice(self.speaker_data[second_speaker])
                
        # Return value       
        target_verify_emb = torch.from_numpy(self.verify_emb[target]).squeeze().to(self.device)
        second_verify_emb = torch.from_numpy(self.verify_emb[second]).squeeze().to(self.device)
        label_type = torch.tensor(label_type, dtype=torch.float, device= self.device)
                    
                    
        return target_verify_emb, second_verify_emb, label_type
    
class TrainingAASIST(Dataset) :
    def __init__(self, aasist_embeddings, speaker_data) :

        self.aasist_emb = aasist_embeddings
        self.speaker_data = speaker_data # a dictionary, each key is the speaker id, value is a list contain the path to the wav file.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) :
        return len(self.aasist_emb.keys())
    
    def __getitem__(self, index) :
        # Randomly create a label first, and then create data base on label_type

        label_type = random.choice([1, 0])
        if label_type == 1 : # which means 2 data files are from 1 person
            
            # Ensure that the speaker has at least 2 bonafide files
            speaker = random.choice(list(self.speaker_data.keys())) # random choice a speaker id
            target, second = random.sample(self.speaker_data[speaker]["bonafide"], 2)
            
        elif label_type == 0 :
            while True :
                speaker = random.choice(list(self.speaker_data.keys()))
                if len(self.speaker_data[speaker]["spoofed_replay"]) + len(self.speaker_data[speaker]["spoofed_voice_clone"]) > 0 :
                    break
            target = random.choice(self.speaker_data[speaker]["bonafide"])
            spoof_list = self.speaker_data[speaker]["spoofed_replay"] + self.speaker_data[speaker]["spoofed_voice_clone"]
            second = random.choice(spoof_list)
            
                
                
        # Return value       
        target_aasist_emb = torch.from_numpy(self.aasist_emb[target]).squeeze().to(self.device)
        second_aasist_emb = torch.from_numpy(self.aasist_emb[second]).squeeze().to(self.device)
        label_type = torch.tensor(label_type, dtype=torch.float, device= self.device)
                           
        return target_aasist_emb, second_aasist_emb, label_type
    