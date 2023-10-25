import pickle as pk

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
