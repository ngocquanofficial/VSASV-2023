import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.append(os.getcwd()) # NOQA
class SiameseNetwork(nn.Module):
    def __init__(self) :
        super().__init__()
        self.DNN_hidden_layer = self._make_layers()
        self.sigmoid = nn.Sigmoid()

        
    def forward_one(self, x) :
        
        x = self.DNN_hidden_layer(x)
        return x

    def _make_layers(self, in_dim= 160, l_nodes= [160, 160, 160]):
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(torch.nn.Linear(in_features = in_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(torch.nn.Linear(in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(torch.nn.LeakyReLU(negative_slope = 0.1))
        return torch.nn.Sequential(*l_fc)
        

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
        pos = (1-label) * torch.pow(cosine_similarity, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive