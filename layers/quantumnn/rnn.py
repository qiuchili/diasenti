# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn
from .activation2 import QActivation2

class QRNNCell(torch.nn.Module):
    def __init__(self, embed_dim, device = torch.device('cpu')):
        super(QRNNCell, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.unitary_x = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).to(self.device),torch.zeros(embed_dim, embed_dim).to(self.device)],dim = -1))
        self.unitary_h = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).to(self.device),torch.zeros(embed_dim, embed_dim).to(self.device)],dim = -1))
        self.Lambda = torch.nn.Parameter(torch.tensor([0.5]))
        
        
    def forward(self, x, h_0):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
        
        if not isinstance(h_0, list):
            raise ValueError('h should be called '
                             'on a list of 2 inputs.')

        if len(h_0) != 2:
            raise ValueError('h should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(h_0)) + ' inputs.')
               
        input_val = self.evolution(x, self.unitary_x)
        hidden_val = self.evolution(h_0, self.unitary_h)
        
        
        output = [ torch.sigmoid(self.Lambda) * _input + (1-torch.sigmoid(self.Lambda)) * _hidden for 
                  _input, _hidden in zip(input_val, hidden_val)]
        
        return output
    
    def evolution(self, x, U):
        #UxU*
        
        x_real = x[0]
        x_imag = x[1]
        
        U_real = U[:,:,0]
        U_imag = U[:,:,1]
        output_real = []
        output_imag = []
        for _x_real, _x_imag in zip(x_real, x_imag):
            
            _r = torch.mm(U_real, _x_real) - torch.mm(U_imag, _x_imag)
            _i = torch.mm(U_imag, _x_real) + torch.mm(U_real, _x_imag)
            
            _output_real = torch.mm(_r, U_real.t()) + torch.mm(_i, U_imag.t())
            _output_imag = torch.mm(_i, U_real.t()) -torch.mm(_r, U_imag.t()) 
            output_real.append(_output_real)
            output_imag.append(_output_imag)
        
        output_real = torch.stack(output_real, dim = 0)
        output_imag = torch.stack(output_imag, dim = 0)
        
        return output_real, output_imag
        
        
        