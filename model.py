import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
class Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.DNN_hidden_layer = self._make_layers()
        self.fc_output = torch.nn.Linear(64, 32, bias= False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, target_verify, second_verify, second_spoof) :
        
        x = torch.cat([target_verify, second_verify, second_spoof], dim= 1)
        x = self.DNN_hidden_layer(x)
        x = self.fc_output(x) # shape (batchsize, 32)
        # x = self.sigmoid(x)
        return x
        

    def _make_layers(self, in_dim= 544, l_nodes= [512, 256, 128, 64]):
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

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # print(losses)
        # print(f"Positive: {distance_positive}, Negative: {distance_negative}")
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
