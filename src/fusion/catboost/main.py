import os
import sys 
sys.path.append(os.getcwd()) # NOQA
import numpy as np
import argparse
import torch


# From ngocquan with love
from src.naive_dnn.utils import compute_eer,load_embeddings,load_pickle
from src.fusion.train import train
def main(args):
    ecapa = args.ecapa_embedding
    s2pecnet = args.s2pecnet_embedding
    lcnn_stft = args.lcnn_stft
    lcnn_cqt = args.lcnn_cqt
    

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
        "--type",
        dest="type",
        type=str,
        help="string, only receive value stft or cqt",
        default="stft",
    )
    parser.add_argument(
        "--ecapa_embedding",
        dest="ecapa_embedding",
        type=str,
        help="path to ecapa embedding",
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
