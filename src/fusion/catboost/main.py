import os
import sys 
sys.path.append(os.getcwd()) # NOQA
import numpy as np
import argparse
import torch


# From ngocquan with love
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
from src.fusion.catboost.dataloader import sample_data
from src.fusion.catboost.train import train
def main(args):
    ecapa = load_pickle(args.ecapa_embedding)
    s2pecnet = load_pickle(args.s2pecnet_embedding)
    aasist = load_pickle(args.aasist_embedding)
    lcnn_stft = load_pickle(args.lcnn_stft)
    lcnn_cqt = load_pickle(args.lcnn_cqt)
    train_dataset = load_pickle(args.train_dataset)
    validation_dataset = load_pickle(args.validation_dataset)
    
    X_train, y_train = sample_data(ecapa, s2pecnet, lcnn_stft, lcnn_cqt, aasist, train_dataset)
    X_test, y_test = sample_data(ecapa, s2pecnet, lcnn_stft, lcnn_cqt, aasist, validation_dataset)

    train(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Detection from Lab914")
    parser.add_argument(
        "--train_dataset",
        dest="train_dataset",
        type=str,
        help="",
        default="path",
    )
    parser.add_argument(
        "--validation_dataset",
        dest="validation_dataset",
        type=str,
        help="",
        default="path",
    )

    parser.add_argument(
        "--ecapa_embedding",
        dest="ecapa_embedding",
        type=str,
        help="path to ecapa embedding",
        default="path",
    )
    parser.add_argument(
        "--aasist_embedding",
        dest="aasist_embedding",
        type=str,
        help="path to aasist embedding",
        default="path",
    )
    parser.add_argument(
        "--s2pecnet_embedding",
        dest="s2pecnet_embedding",
        type=str,
        help="path to s2pecnet embedding",
        default="path",
    )

    parser.add_argument(
        "--lcnn_stft",
        dest="lcnn_stft",
        type=str,
        help="path to lcnn embedding with stft input",
        default="path",
    )

    parser.add_argument(
        "--lcnn_cqt",
        dest="lcnn_cqt",
        type=str,
        help="path to lcnn embedding with cqt input",
        default="path",
    )

    main(parser.parse_args())
