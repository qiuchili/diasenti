# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QActivation(torch.nn.Module):
    def __init__(self, scale_factor=1, beta=0.5):
        super(QActivation, self).__init__()
        self.scale_factor = scale_factor
        self.beta = beta
     
    def forward(self, x):

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
    
        
        x_real = x[0]
        x_imag = x[1]
        
        diagonal_values = torch.stack([torch.diag(_real) for _real in x_real],dim=0)
        
        diagonal_values = F.softmax(diagonal_values*self.scale_factor, dim = -1)
        
        diagonal_mats = [torch.diag(_diagonal) for _diagonal in diagonal_values]
        
        output_real = [ _real*self.beta+ (1-self.beta)*_mat for _real,_mat in zip(x_real, diagonal_mats)]
        output_r = torch.stack(output_real, dim=0)
        
        output_i = x_imag * self.beta
        
        
        return [output_r, output_i]
    
#
        


    
    
    
    
