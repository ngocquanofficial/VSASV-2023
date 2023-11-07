import os
import sys 
sys.path.append(os.getcwd()) # NOQA
import torch

class Maxout(torch.nn.Module) :
    def __init__(self, num_units: int, axis: int = 1, **kwargs) :
        super().__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def forward(self, input) :
        # Check the inputs type is numpy or torch
        if not torch.is_tensor(input) :
            input = torch.from_numpy(input)
            
        shape = list(input.shape) # get shape as list
        num_channels = shape[self.axis]
            
        if self.axis < 0 :
            axis = self.axis + len(shape)
        else :
            axis = self.axis
            
        # copy the list shape
        extended_shape = shape[:]
        extended_shape[axis] = self.num_units
        k = num_channels // self.num_units
        extended_shape.insert(axis, k) 
        
        # Change from (batchsize,..., num_channels,...) -> (batchsize, ..., num_channels//num_units, num_units,...)
        reshape_input = torch.reshape(input, extended_shape)
        max_values, max_indices = torch.max(reshape_input, axis, keepdim= False)
        output = max_values # only receive the max, not the max_indices
        
        return output
    
    def get_config(self) :
        config = {"num_units": self.num_units, "axis": self.axis}
        return config
    
            
            
            
            
        
            
    