import random
import torch
from torch.utils.data import Dataset
# use the embedding vector implemented from 2 model AASIST(antispoof-model) and ECAPA(verification-model)
class TrainingVLSPDataset(Dataset) :
    def __init__(self, antispoof_file, verification_file, speaker_data) :
        """

        Args:
            antispoof_file (dictioinary): a dictionary, key-value pair is filename-embedding vector created by anti-spoofing model
            verification_file (dictionary): _description_
            speaker_data (dictionary): a dict with key= speaker id, values is a dict contain 2 sub-list, each list contains path to bonafine and fake voices respectively.
        """
        self.antispoof_emb = antispoof_file
        self.verify_emb = verification_file
        self.speaker_data = speaker_data
    
    def __len__(self) :
        return len(self.verify_emb.keys())
    
    def __getitem__(self, index) :
        # Randomly create a label first, and then create data base on label_type
        """_summary_

        Args:
            index (_type_): _description_
            
            target, second are two wav file path,
        Return: 
        self.verify_emb[target]: speaker verification embedding of the target wav file
        self.verify_emb[second]: speaker verification embedidng of the second wav file
        self.antispoof_emb[second]: anti-spoofing embedding of the second wav file
        label_type: the label corresponding to 2 wav files, = 1 if two files are both bonafile, from the same person

        """
        label_type = random.randint(0, 1)
        if label_type == 1 : # which means 2 data files are both bonafine, from 1 person
            speaker = random.choice(list(self.speaker_data.keys())) # random choice a speaker id
            target, second = random.sample(self.speaker_data[speaker]["bonafide"], 2)
            
        elif label_type == 0 :
            second_type = random.randint(1, 2)
            
            if second_type == 1 : # Both 2 file is bonafile, but from different people
                target_speaker, second_speaker = random.sample(self.speaker_data.keys(), 2)
                target = random.choice(self.speaker_data[target_speaker]["bonafine"])
                second = random.choice(self.speaker_data[second_speaker]["bonafine"])
            
            if second_type == 2 : # the second file is a spoofing
                speaker = random.choice(list(self.speaker_data.keys()))
                
                if len(self.speaker_data[speaker]["spoofed_voice_clone"]) + len(self.speaker_data[speaker]["spoofed_replay"]) == 0 :
                    # There is not any spoofing voice for speaker
                    # Then change the speaker until find his/her fake voice
                    
                    while True :
                        speaker = random.choice(list(self.speaker_data.keys()))
                        if len(self.speaker_data[speaker]["spoofed_voice_clone"]) + len(self.speaker_data[speaker]["spoofed_replay"]) > 1 :
                            break
                    
                    # From here, the speaker has at least one spoofing voice
                    
                    # Merge all type of spoofing voice into 1 list, then random choose 1
                    voice_spoofing_list = self.speaker_data[speaker]["spoofed_voice_clone"] + self.speaker_data[speaker]["spoofed_replay"]
                    
                    target = random.choice(self.speaker_data[speaker]["bonafine"])
                    second = random.choice(voice_spoofing_list)
                    
        return self.verify_emb[target], self.verify_emb[second], self.antispoof_emb[second], label_type                    
                
        
        