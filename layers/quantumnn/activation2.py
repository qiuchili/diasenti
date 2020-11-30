# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QActivation2(torch.nn.Module):
    def __init__(self, scale_factor=1):
        super(QActivation2, self).__init__()
        self.scale_factor = scale_factor
     
    def forward(self, x):

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
    
        
        # Apply softmax on diagonal values
        # and scale the off-diagonal values to ensure the outcome is still a density matrix
        x_real = x[0]
        x_imag = x[1]
        
        diagonal_values = torch.diagonal(x_real.permute(1,2,0))
        new_diagonal_values = F.softmax(diagonal_values*self.scale_factor, dim = -1)
        
        # To scale the off-diagonal values 
        max_ratio = torch.max(diagonal_values/new_diagonal_values, dim = -1).values
        
        x_real = x_real/max_ratio.view(len(max_ratio),1,1) 
        x_imag = x_imag/max_ratio.view(len(max_ratio),1,1)
        
        x_real = [l.fill_diagonal_(0) for l in x_real]
        x_real = torch.stack(x_real,dim = 0)
        x_real = x_real + torch.diag_embed(new_diagonal_values)
        
        x_imag = [l.fill_diagonal_(0) for l in x_imag]
        x_imag = torch.stack(x_imag,dim = 0)
            
        
        return [x_real, x_imag]
    
#
        


    
    
    
    
