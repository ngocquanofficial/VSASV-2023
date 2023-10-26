import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from dataloader import TrainingVLSPDataset
from torch.utils.data import DataLoader
from model import Model
import torch
import torch.optim as optim
# From ngocquan with love
from utils import *
from train import train
from utils import load_pickle
def main(args):

    mode = args.mode
    if mode != "train" and mode != "test" :
        print("Type the mode equals train or test, run again")
        # Fast return
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    antispoof_file = load_pickle(args.aasist_embedding)
    verification_file = load_pickle(args.ecapa_embedding)
    speaker_data = load_pickle(args.speaker_embedding)
    
    training_data = TrainingVLSPDataset(antispoof_file= antispoof_file, verification_file= verification_file, speaker_data= speaker_data)
    train_loader = DataLoader(dataset= training_data, batch_size= 4, shuffle= True)
    model = Model().to(device) 
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
    if mode == "train" :
        train(model= model, optimizer= optimizer, criterion= criterion, data_loader= train_loader, num_epochs= 50)
    
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
    main(parser.parse_args())