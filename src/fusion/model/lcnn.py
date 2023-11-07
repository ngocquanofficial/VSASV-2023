import os
import sys 
sys.path.append(os.getcwd()) # NOQA

from src.fusion.model.layers import Maxout
import torch
import torch.nn as nn
from torch.nn import Conv2d

class LCNN(nn.Module) :
    def __init__(self, input_dim, num_label) :
        super().__init__()
    
        self.input_dim = input_dim
        self.num_label = num_label
        self.lcnn = nn.Sequential(
            self._make_maxout_conv(input_dim, 64, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            self._make_maxout_conv(input_channels= 32, output_channels= 64, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(32),

            self._make_maxout_conv(input_channels= 32, output_channels= 96, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(48),

            self._make_maxout_conv(input_channels= 48, output_channels= 96, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(48),

            self._make_maxout_conv(input_channels= 48, output_channels= 128, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            self._make_maxout_conv(input_channels= 64, output_channels= 128, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(64),

            self._make_maxout_conv(input_channels= 64, output_channels= 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),

            self._make_maxout_conv(input_channels= 32, output_channels= 64, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(32),

            self._make_maxout_conv(input_channels= 32, output_channels= 64, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.linear = nn.Sequential(
            self._make_maxout_dense(output_dim= 160),
            nn.BatchNorm1d(80),
            nn.Dropout(p= 0.75),
            nn.Linear(in_features= 80, out_features= self.num_label)
        )
        
    def forward(self, input) :
        x = self.lcnn(input)
        
        x = x.flatten(1)
        x = self.linear(x)
        return x
    
    def _make_maxout_conv(
        self, 
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        padding: str = "same"
    ):
        return nn.Sequential(
            Conv2d(
                in_channels= input_channels,
                out_channels= output_channels,
                kernel_size= kernel_size,
                stride= stride, 
                padding= padding),
            Maxout(num_units= output_channels//2)
        )

    def _make_maxout_dense(
        self,
        output_dim: int
    ) :
        return nn.Sequential(
            nn.LazyLinear(out_features=output_dim),
            Maxout(output_dim//2, axis= 1)
        )
    
# x = torch.rand((100, 3, 100, 100))
# model = LCNN(input_dim= 3, num_label= 2)
# print(model(x).shape)

# from torchsummary import summary

# summary(model, ( 3, 100, 100))