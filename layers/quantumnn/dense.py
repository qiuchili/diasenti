# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QDense(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device = torch.device('cpu')):
        super(QDense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.weight = nn.Parameter(torch.stack([torch.eye(input_dim).to(self.device),torch.zeros(input_dim, input_dim).to(self.device)],dim = -1))
        self._rand = nn.Parameter(torch.rand(self.output_dim)).to(self.device)
        self.Lambda = nn.Parameter(torch.tensor([0.5])).to(self.device)

        
    def forward(self, x):

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
    
        input_val = self.evolution(x, self.weight)
        
        if self.input_dim >= self.output_dim:       
            sub_matrix_r = [ _[:self.output_dim, :self.output_dim] for _ in input_val[0]]
            
            sub_matrix_i = [ _[:self.output_dim, :self.output_dim] for _ in input_val[1]]
            
            _r = torch.stack([ mat_r/sum(torch.diag(mat_r)) for mat_r in sub_matrix_r],dim=0)
            _i = torch.stack([ mat_i/sum(torch.diag(mat_r)) for mat_i,mat_r in zip(sub_matrix_i,sub_matrix_r)],dim=0)
            sub_matrix = [_r, _i]
        
        else:    
            sub_matrix = [torch.zeros(len(x[0]),self.output_dim,self.output_dim).to(self.device)]* 2         
            sub_matrix[0][:, :self.input_dim, :self.input_dim] = input_val[0]
            sub_matrix[1][:, :self.input_dim, :self.input_dim] = input_val[1]
         
        normalized_rand = F.softmax(self._rand,dim=-1)
        output_r = torch.sigmoid(self.Lambda) * sub_matrix[0] + (1-torch.sigmoid(self.Lambda)) * torch.diag(normalized_rand)
        output_i = torch.sigmoid(self.Lambda) * sub_matrix[1]
        return [output_r, output_i]
    
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
#
        


    
    
    
    
