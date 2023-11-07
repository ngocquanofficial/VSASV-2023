import random
import torch
from torch.utils.data import Dataset
# use the embedding vector implemented from 2 model AASIST(antispoof-model) and ECAPA(verification-model)
class TrainingVLSPDataset(Dataset) :
    def __init__(self, antispoof_embeddings, verification_embeddings, speaker_data) :
        """

        Args:
            antispoof_file (dictioinary): a dictionary, key-value pair is filename-embedding vector created by anti-spoofing model
            verification_file (dictionary): _description_
            speaker_data (dictionary): a dict with key= speaker id, values is a dict contain 2 sub-list, each list contains path to bonafide and fake voices respectively.
        """
        self.antispoof_emb = antispoof_embeddings
        self.verify_emb = verification_embeddings
        self.speaker_data = speaker_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        label_type: the label corresponding to 2 wav files, = 1 if two files are both bonafide, from the same person

        """
        label_type = random.randint(0, 1)
        if label_type == 1 : # which means 2 data files are both bonafide, from 1 person
            
            # Ensure that the speaker has at least 2 bonafide files
            speaker = random.choice(list(self.speaker_data.keys())) # random choice a speaker id
            target, second = random.sample(self.speaker_data[speaker]["bonafide"], 2)
            
        elif label_type == 0 :
            second_type = random.randint(1, 2)
            
            if second_type == 1 : # Both 2 file is bonafide, but from different people
                target_speaker, second_speaker = random.sample(self.speaker_data.keys(), 2)
           
                target = random.choice(self.speaker_data[target_speaker]["bonafide"])
                second = random.choice(self.speaker_data[second_speaker]["bonafide"])
            
            elif second_type == 2 : # the second file is a spoofing
                speaker = random.choice(list(self.speaker_data.keys()))
                # Sometime speaker do not have both spoofed_voice_clone and spoofed_replay, so we have a solution:
                
                if len(self.speaker_data[speaker]["spoofed_voice_clone"]) + len(self.speaker_data[speaker]["spoofed_replay"]) == 0 :
                    # There is not any spoofing voice for speaker
                    # Then change the speaker until find his/her fake voice
                    
                    while True :
                        speaker = random.choice(list(self.speaker_data.keys()))
                        if len(self.speaker_data[speaker]["spoofed_voice_clone"]) + len(self.speaker_data[speaker]["spoofed_replay"]) > 0 :
                            break
                    # End the if statement to find speaker
                    
                # From here, the speaker has at least one spoofing voice
                
                # Merge all type of spoofing voice into 1 list, then random choose 1
                voice_spoofing_list = self.speaker_data[speaker]["spoofed_voice_clone"] + self.speaker_data[speaker]["spoofed_replay"]
                
                target = random.choice(self.speaker_data[speaker]["bonafide"])
                second = random.choice(voice_spoofing_list)
                
        # Return value       
        target_verify_emb = torch.from_numpy(self.verify_emb[target]).squeeze().to(self.device)
        second_verify_emb = torch.from_numpy(self.verify_emb[second]).squeeze().to(self.device)
        second_antispoof_emb = torch.from_numpy(self.antispoof_emb[second]).squeeze().to(self.device)
        label_type = torch.tensor(label_type, dtype=torch.float, device= self.device)
                    
                    
        return target_verify_emb, second_verify_emb, second_antispoof_emb, label_type
                
class TrainingVLSPDatasetWithTripleLoss(Dataset) :
    def __init__(self, antispoof_embeddings, verification_embeddings, speaker_data) :
        """

        Args:
            antispoof_file (dictioinary): a dictionary, key-value pair is filename-embedding vector created by anti-spoofing model
            verification_file (dictionary): _description_
            speaker_data (dictionary): a dict with key= speaker id, values is a dict contain 2 sub-list, each list contains path to bonafide and fake voices respectively.
        """
        self.antispoof_emb = antispoof_embeddings
        self.verify_emb = verification_embeddings
        self.speaker_data = speaker_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self) :
        return len(self.verify_emb.keys())
    
    def __getitem__(self, index) :
            
        # Ensure that the speaker has at least 2 bonafide files by preprocessing data
        speaker = random.choice(list(self.speaker_data.keys())) # random choice a speaker id
        target, positive_utterance = random.sample(self.speaker_data[speaker]["bonafide"], 2)
        negative_type = random.randint(0, 1)
        if negative_type == 0 :
            # same speaker, but spoof utterance          
            if len(self.speaker_data[speaker]["spoofed_voice_clone"]) + len(self.speaker_data[speaker]["spoofed_replay"]) == 0 :
                negative_type = 1
                # continue sample the next case
            else :
                voice_spoofing_list = self.speaker_data[speaker]["spoofed_voice_clone"] + self.speaker_data[speaker]["spoofed_replay"]
                negative_utterance = random.choice(voice_spoofing_list)
                
        elif negative_type == 1 :
            # different speaker
            second_speaker = random.choice(list(self.speaker_data.keys()))
            if second_speaker == speaker :
                second_speaker = random.choice(list(self.speaker_data.keys()))
                
            second_voice_list = self.speaker_data[second_speaker]["spoofed_voice_clone"] + self.speaker_data[second_speaker]["spoofed_replay"] + self.speaker_data[second_speaker]["bonafide"]
            negative_utterance = random.choice(second_voice_list)
            
        anchor_tuple = self.get_embedding(target)
        positive_tuple = self.get_embedding(positive_utterance)
        negative_tuple = self.get_embedding(negative_utterance)
        
        return anchor_tuple, positive_tuple, negative_tuple
                
    def get_embedding(self, target) : 
        # Return value       
        target_verify_emb = torch.from_numpy(self.verify_emb[target]).squeeze().to(self.device)
        target_antispoof_emb = torch.from_numpy(self.antispoof_emb[target]).squeeze().to(self.device)

                    
                    
        return (target_verify_emb, target_antispoof_emb, target_antispoof_emb)
                
    
