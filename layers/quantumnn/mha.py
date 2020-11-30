# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn
import torch.nn.functional as F
from optimizer.vanilla_unitary import Vanilla_Unitary

class QMHA(torch.nn.Module):
    def __init__(self, embed_dim, device = torch.device('cpu')):
        super(QMHA, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.unitary_q = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1)).to(self.device)
        self.unitary_k = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1)).to(self.device)
        self.unitary_v = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1)).to(self.device)


    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

     
        # list of [real, imag]
        # each array is of shape (batch_size, embed_dim, embed_dim)
        queries = [self.evolution(input_t, self.unitary_q) for input_t in inputs]
        keys = [self.evolution(input_t, self.unitary_k) for input_t in inputs]

        values = [self.evolution(input_t, self.unitary_v) for input_t in inputs]
        
        results = []
        for _q in queries:
            scores = torch.tensor([ self.distance(_q, _k) for _k in keys]).to(self.device)
            scores = F.softmax(scores,dim=-1)
            
            # scores shape: (time_stamps,batch_size)
            result_q = [torch.zeros_like(inputs[0][0]), torch.zeros_like(inputs[0][0])]
            for v_r, v_i, score in zip(values[0],values[1],scores):
                result_q[0] = result_q[0] + v_r* score.unsqueeze(dim = -1).unsqueeze(dim = -1).expand_as(v_r)
                result_q[1] = result_q[1] + v_i* score.unsqueeze(dim = -1).unsqueeze(dim = -1).expand_as(v_i)
                
            results.append(result_q)
                
    
        return results
    
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
            
            
    #        U_hermitian_real = U_real.t()
    #        U_hermitian_imag = -U_imag.t()
            _output_real = torch.mm(_r, U_real.t()) + torch.mm(_i, U_imag.t())
            _output_imag = torch.mm(_i, U_real.t()) - torch.mm(_r, U_imag.t()) 
            output_real.append(_output_real)
            output_imag.append(_output_imag)
        
        output_real = torch.stack(output_real, dim = 0)
        output_imag = torch.stack(output_imag, dim = 0)

        
        return output_real, output_imag
        

    #Trace Inner Product
    def distance(self, query, key):
        results_real = torch.matmul(query[0], key[0]) - torch.matmul(query[1], key[1])
        results_imag = torch.matmul(query[0], key[1]) + torch.matmul(query[1], key[0])
        res = [(torch.trace(x)**2 + torch.trace(y)**2)**0.5 for x,y in zip(results_real, results_imag)]
        return res
    
    
