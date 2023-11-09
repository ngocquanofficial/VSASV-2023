import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.fusion.dataloader import TrainingDataLCNN, ValidationDataLCNN
from src.fusion.LCNN.model.lcnn import LCNN

# From ngocquan with love
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
from src.fusion.train import train
def main(args):

    mode = args.mode
    if mode != "train" and mode != "test" :
        print("Type the mode equals train or test, run again")
        # Fast return
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCNN(input_dim=0, num_label= 2).to(device) 
    
    training_file = load_pickle(args.training_file)
    validation_file = load_pickle(args.validation_file)
    
    training_data = TrainingDataLCNN(path_list= training_file)
    validation_data = TrainingDataLCNN(path_list= validation_file)
    validation_loader = DataLoader(dataset= validation_data, batch_size= 1, shuffle= False)
    train_loader = DataLoader(dataset= training_data, batch_size= 32, shuffle= True)
    criterion = torch.nn.CrossEntropyLoss()
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
        "--training_file",
        dest="training_file",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--validation_file",
        dest="validation_file",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    main(parser.parse_args())