import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.fusion.dataloader import TrainingDataLCNN, ValidationDataLCNN, GenEmbDataLCNN
from src.fusion.LCNN.model.lcnn import LCNN

from tqdm import tqdm
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
import pickle as pk


def main(args):

    mode = args.mode
    type_data = args.type
    if mode != "train" and mode != "test" :
        print("Type the mode equals train or test, run again")
        # Fast return
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCNN(input_dim= 1, num_label= 2).to(device)
    model = torch.load(args.ckpt)
    model.eval()
    
    dictionary_file = load_pickle(args.dictionary_file)
    stft_embedding = load_pickle(args.stft_embedding) 
    cqt_embedding = load_pickle(args.cqt_embedding)
    mel_embedding = load_pickle(args.mel_embedding)
    
    
    dataset = GenEmbDataLCNN(path_list= dictionary_file, stft_embedding= stft_embedding, cqt_embedding= cqt_embedding, mel_embedding= mel_embedding, type= type_data)
    data_loader = DataLoader(dataset= dataset, batch_size= 32, shuffle= True)
    
    last_hidden_dict = {}
    output_dict = {}

    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader)) :
            wave, label, audio_paths = data
            last_hidden, output = model(wave)
            last_hidden = last_hidden.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            for v, audio_path in zip(last_hidden, audio_paths):
                # relative_path = "/".join(audio_path.split("/")[-4:])
                last_hidden_dict[audio_path] = v
            for v, audio_path in zip(output, audio_paths):
                # relative_path = "/".join(audio_path.split("/")[-4:])
                output_dict[audio_path] = v
    
    with open(f"/kaggle/working/embeddings_{type_data}_{mode}.pkl", "wb") as f:
        pk.dump(last_hidden_dict, f)
    with open(f"/kaggle/working/scores_{type_data}_{mode}.pkl", "wb") as f:
        pk.dump(output_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unknown Team from Lab914")
    
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--type",
        dest="type",
        type=str,
        help="string, only receive value stft or cqt",
        default="train",
    )

    parser.add_argument(
        "--ckpt",
        dest="ckpt",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--stft_embedding",
        dest="stft_embedding",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--cqt_embedding",
        dest="cqt_embedding",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--mel_embedding",
        dest="mel_embedding",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )
    parser.add_argument(
        "--dictionary_file",
        dest="dictionary_file",
        type=str,
        help="",
        default="Dien di dung luoi :) ",
    )

    main(parser.parse_args())