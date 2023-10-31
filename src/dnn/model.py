import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
class Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.DNN_hidden_layer = self._make_layers()
        self.fc_output = torch.nn.Linear(256, 128, bias= False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, target_verify, second_verify, second_spoof) :
        
        x = torch.cat([target_verify, second_verify, second_spoof], dim= 1)
        x = self.DNN_hidden_layer(x)
        x = self.fc_output(x) # shape (batchsize, 128)
        return x

    def _make_layers(self, in_dim= 544, l_nodes= [512, 480, 256]):
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
    
