import pickle as pk
import sklearn
from sklearn.metrics import roc_curve
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def load_embeddings(self):
    # load saved countermeasures(CM) related preparations
    with open(self.config.dirs.embedding + "cm_embd_trn.pk", "rb") as f:
        self.cm_embd_trn = pk.load(f)
    with open(self.config.dirs.embedding + "cm_embd_dev.pk", "rb") as f:
        self.cm_embd_dev = pk.load(f)
    with open(self.config.dirs.embedding + "cm_embd_eval.pk", "rb") as f:
        self.cm_embd_eval = pk.load(f)

    # load saved automatic speaker verification(ASV) related preparations
    with open(self.config.dirs.embedding + "asv_embd_trn.pk", "rb") as f:
        self.asv_embd_trn = pk.load(f)
    with open(self.config.dirs.embedding + "asv_embd_dev.pk", "rb") as f:
        self.asv_embd_dev = pk.load(f)
    with open(self.config.dirs.embedding + "asv_embd_eval.pk", "rb") as f:
        self.asv_embd_eval = pk.load(f)

    # load speaker models for development and evaluation sets
    with open(self.config.dirs.embedding + "spk_model_dev.pk", "rb") as f:
        self.spk_model_dev = pk.load(f)
    with open(self.config.dirs.embedding + "spk_model_eval.pk", "rb") as f:
        self.spk_model_eval = pk.load(f)

def load_pickle(url) :
    with open(url, 'rb') as file:
        loaded_data = pk.load(file)
    return loaded_data

def save_pickle(data, filename= "ngocquan.pk") :

    # Store data (serialize)
    with open(filename, 'wb') as handle:
        pk.dump(data, handle)

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred)

    fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
    sasv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return sasv_eer



# label = [1,1,1,1,1,1,0,0]
# pred = [1,1,1,1,1,1,1,0]
# print(compute_eer(label, pred))

