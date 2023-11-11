import torch
import numpy as np
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
import random
import os


def sample_data(ecapa_emb, s2pecnet_emb, lcnn_stft_emb, lcnn_cqt_emb, aasist_emb, dataset) :
    X_data = []
    y_label = []

    for data in dataset :
        target, second, label = data
        emb_concat = np.concatenate((ecapa_emb[target], ecapa_emb[second], aasist_emb[second], s2pecnet_emb[second], lcnn_stft_emb[second], lcnn_cqt_emb[second]), axis= 0)
   
        X_data.append(emb_concat)
        y_label.append(label)
    
    return np.array(X_data), np.array(y_label)
            


def sample_data_from_scratch(ecapa_file, s2pecnet_file, aasist_file, lcnn_stft_file, lcnn_cqt_file, speaker_data, num= 10000) :
    X_data = []
    y_label = []
    ecapa_emb = load_pickle(ecapa_file)
    s2pecnet_emb = load_pickle(s2pecnet_file)
    lcnn_stft_emb = load_pickle(lcnn_stft_file)
    lcnn_cqt_emb = load_pickle(lcnn_cqt_file)
    speaker_data = load_pickle(speaker_data)
    aasist_data = load_pickle(aasist_file)

    for idx in range(num) :

        label_type = random.randint(0, 1)
        if label_type == 1 : # which means 2 data files are both bonafide, from 1 person
            
            # Ensure that the speaker has at least 2 bonafide files
            speaker = random.choice(list(speaker_data.keys())) # random choice a speaker id
            target, second = random.sample(speaker_data[speaker]["bonafide"], 2)
            
        elif label_type == 0 :
            second_type = random.randint(1, 2)
            
            if second_type == 1 : # Both 2 file is bonafide, but from different people
                target_speaker, second_speaker = random.sample(speaker_data.keys(), 2)
           
                target = random.choice(speaker_data[target_speaker]["bonafide"])
                second = random.choice(speaker_data[second_speaker]["bonafide"])
            
            elif second_type == 2 : # the second file is a spoofing
                speaker = random.choice(list(speaker_data.keys()))
                # Sometime speaker do not have both spoofed_voice_clone and spoofed_replay, so we have a solution:
                
                if len(speaker_data[speaker]["spoofed_voice_clone"]) + len(speaker_data[speaker]["spoofed_replay"]) == 0 :
                    # There is not any spoofing voice for speaker
                    # Then change the speaker until find his/her fake voice
                    
                    while True :
                        speaker = random.choice(list(speaker_data.keys()))
                        if len(speaker_data[speaker]["spoofed_voice_clone"]) + len(speaker_data[speaker]["spoofed_replay"]) > 0 :
                            break
                    # End the if statement to find speaker
                    
                # From here, the speaker has at least one spoofing voice
                
                # Merge all type of spoofing voice into 1 list, then random choose 1
                voice_spoofing_list = speaker_data[speaker]["spoofed_voice_clone"] + speaker_data[speaker]["spoofed_replay"]
                
                target = random.choice(speaker_data[speaker]["bonafide"])
                second = random.choice(voice_spoofing_list)           
           
        data = torch.cat(ecapa_emb[target], ecapa_emb[second], s2pecnet_emb[second], lcnn_stft_emb[second], lcnn_cqt_emb[second])
        X_data.append(data.numpy())
        y_label.append(label_type)
        
    return np.array(X_data), np.array(y_label)
            


