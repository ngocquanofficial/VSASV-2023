import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import argparse
from src.detect_speaker.dataloader import VietnamCeleb, TrainingAASIST
from torch.utils.data import DataLoader
from src.detect_speaker.model import SiameseNetwork, ContrastiveLoss
import torch
import torch.optim as optim
# From ngocquan with love
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
from src.detect_speaker.train import train
def main(args):

    mode = args.mode
    if mode != "train" and mode != "test" :
        print("Type the mode equals train or test, run again")
        # Fast return
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device) 
    
    aasist_embeddings = load_pickle(args.aasist_embedding)
    speaker_data = load_pickle(args.speaker_embedding)
    
    training_data = TrainingAASIST(aasist_embeddings= aasist_embeddings, speaker_data= speaker_data)
    validation_data = TrainingAASIST(aasist_embeddings= aasist_embeddings, speaker_data= speaker_data)
    validation_loader = DataLoader(dataset= validation_data, batch_size= 1, shuffle= False)
    train_loader = DataLoader(dataset= training_data, batch_size= 32, shuffle= True)
    criterion = ContrastiveLoss(margin= 1)
    optimizer = optim.AdamW(model.parameters(), lr= 1e-5)
    
    if mode == "train" :
        train(model= model, optimizer= optimizer, criterion= criterion, data_loader= train_loader, num_epochs= 50, validation_loader= validation_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Detection from Lab914")

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        help="string, only receive value train or test, depend on training or testing",
        default="train",
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
        "--aasist_embedding",
        dest="aasist_embedding",
        type=str,
        help="path to the pickle file containing aasist embeddings",
        default="Dien di dung luoi :) ",
    )
    main(parser.parse_args())