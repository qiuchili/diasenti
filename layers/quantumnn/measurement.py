# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn

class QMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, device = torch.device('cpu')):
        super(QMeasurement, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.kernel = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).to(self.device),torch.zeros(embed_dim, embed_dim).to(self.device)],dim = -1))


    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
    
    
        input_real = inputs[0]
        input_imag = inputs[1]
        
        real_kernel = self.kernel[:,:,0]
        imag_kernel = self.kernel[:,:,1]
       
        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)


        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        output_real = torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t())\
            - torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1), torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
    
        return output_real

    
    
    
    
