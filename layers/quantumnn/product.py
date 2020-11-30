# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class QProduct(torch.nn.Module):

    def __init__(self, device = torch.device('cuda')):
        super(QProduct, self).__init__()
        self.device = device

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list.')
            
        batch_size = inputs[0][0].shape[0]
        seq_len = inputs[0][0].shape[1]
        real_tensors = []
        imag_tensors = []
        
        for i in range(seq_len):   
            tensor_product_real = torch.ones(batch_size,1).to(self.device)
            tensor_product_imag = torch.ones(batch_size,1).to(self.device)
            
            for h_real, h_imag in inputs:
               
                h_added_real = h_real[:,i,:]
                h_added_imag = h_imag[:,i,:]
                result_real = torch.bmm(tensor_product_real.unsqueeze(2),h_added_real.unsqueeze(1))-torch.bmm(tensor_product_imag.unsqueeze(2),h_added_imag.unsqueeze(1))
                
                result_imag = torch.bmm(tensor_product_real.unsqueeze(2),h_added_imag.unsqueeze(1)) +torch.bmm(tensor_product_imag.unsqueeze(2),h_added_real.unsqueeze(1)) 
                
                tensor_product_real = result_real.view(batch_size,-1)
                tensor_product_imag = result_imag.view(batch_size,-1)
                
            real_tensors.append(tensor_product_real)
            imag_tensors.append(tensor_product_imag)
        
        real_states = torch.stack(real_tensors, dim =1)
        imag_states = torch.stack(imag_tensors, dim =1)

        return [real_states, imag_states]
