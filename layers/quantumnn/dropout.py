# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn

class QDropout(torch.nn.Module):
    def __init__(self, p=0.5, device = torch.device('cpu')):
        super(QDropout, self).__init__()
        self.dropout = p
        self.device = device
        
    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
            
        # Only Support each input having shape (batch_size, embed_dim, embed_dim)
        x_real = x[0]
        x_imag = x[1]
        batch_size = len(x_real)
        dimension = x_real.shape[-1]
        
        binary_ids = torch.bernoulli(torch.ones(batch_size,dimension)*(1-self.dropout)).to(self.device)
        mask_tensor = torch.ones_like(x_real)
        mask_tensor[binary_ids == 0,:] = 0
        temp = mask_tensor.transpose(0,1) 
        temp[:,binary_ids == 0] = 0
        mask_tensor = temp.transpose(0,1)
        output_real = torch.stack([torch.diag(x_r.diagonal()) for x_r in x_real],dim=0)
        output_imag = torch.stack([torch.diag(x_i.diagonal()) for x_i in x_imag],dim=0)
        
        output_real[mask_tensor ==1] = x_real[mask_tensor ==1]
        output_imag[mask_tensor ==1] = x_imag[mask_tensor ==1]

    
        return output_real, output_imag
#
        


    
    
    
    
