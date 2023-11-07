import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from dataloader import TrainingVLSPDataset, TrainingVLSPDatasetWithTripleLoss
from torch.utils.data import DataLoader
from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# From ngocquan with love
from utils import *
from train import train, train_triplet_loss
from utils import load_pickle

def main(args):

    mode = args.mode
    if mode != "train" and mode != "test" :
        print("Type the mode equals train or test, run again")
        # Fast return
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device) 
    
    antispoof_embeddings = load_pickle(args.aasist_embedding)
    verification_embeddings = load_pickle(args.ecapa_embedding)
    speaker_data = load_pickle(args.speaker_embedding)
    
    if args.loss == "mse" :
        training_data = TrainingVLSPDataset(antispoof_embeddings= antispoof_embeddings, verification_embeddings= verification_embeddings, speaker_data= speaker_data)
        train_loader = DataLoader(dataset= training_data, batch_size= 64, shuffle= True)
        criterion = torch.nn.MSELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
        if mode == "train" :
            train(model= model, optimizer= optimizer, criterion= criterion, data_loader= train_loader, num_epochs= 50)
            
    elif args.loss == 'triplet' :
        training_data = TrainingVLSPDatasetWithTripleLoss(antispoof_embeddings= antispoof_embeddings, verification_embeddings= verification_embeddings, speaker_data= speaker_data)
        validation_data = TrainingVLSPDataset(antispoof_embeddings= antispoof_embeddings, verification_embeddings= verification_embeddings, speaker_data= speaker_data)
        validation_loader = DataLoader(dataset= validation_data, batch_size= 1, shuffle= False)
        train_loader = DataLoader(dataset= training_data, batch_size= 32, shuffle= True)
        criterion = nn.TripletMarginWithDistanceLoss(distance_function= lambda x, y: 1.0 - F.cosine_similarity(x, y), margin= 0.9)
        optimizer = optim.AdamW(model.parameters(), lr= 2e-5)
        
        if mode == "train" :
            train_triplet_loss(model= model, optimizer= optimizer, criterion= criterion, data_loader= train_loader, num_epochs= 50, validation_loader= validation_loader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLSP2023 from Lab914")

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        help="string, only receive value train or test, depend on training or testing",
        default="train",
    )
    parser.add_argument(
        "--aasist_embedding",
        dest="aasist_embedding",
        type=str,
        help="path to the pickle file containing aasist embeddings",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--ecapa_embedding",
        dest="ecapa_embedding",
        type=str,
        help="path to the pickle file containing ecapa embeddings",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--speaker_embedding",
        dest="speaker_embedding",
        type=str,
        help="path to the pickle file containing speaker embeddings",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        help="loss function type",
        default="mse",
    )
    main(parser.parse_args())