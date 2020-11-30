# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn
import numpy as np
from layers.complexnn.measurement import ComplexMeasurement

class ComplexMeasurement2(torch.nn.Module):
    def __init__(self, embed_dim, units=5, ortho_init=False, device = torch.device('cpu')):
        super(ComplexMeasurement2, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if ortho_init:
            self.kernel = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).to(device),torch.zeros(embed_dim, embed_dim).to(device)],dim = -1))

        else:
            rand_tensor = torch.rand(self.units, self.embed_dim, 2).to(device)
            normalized_tensor = F.normalize(rand_tensor.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, self.embed_dim, 2)
            self.kernel = torch.nn.Parameter(normalized_tensor)


    def forward(self, inputs, measure_operator=None):
        
        input_real = inputs[0]
        input_imag = inputs[1]
        weights = inputs[2]
        
        if weights.dim() == 4:
            input_real = input_real.transpose(1,2)
            input_imag = input_imag.transpose(1,2)
            weights = weights.transpose(1,2)
            
        real_kernel = self.kernel[:,:,0]
        imag_kernel = self.kernel[:,:,1]
       
        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)
        results = []
        for r_k, i_k in zip(real_kernel,imag_kernel):
            mul_real = (torch.matmul(input_real, r_k)+ torch.matmul(input_imag, i_k))**2
            mul_imag = (torch.matmul(input_imag, r_k)- torch.matmul(input_real, i_k))**2
#            result = torch.matmul(weights.transpose(1,2), mul_real+mul_imag).squeeze()
                
            result = torch.matmul(weights.transpose(-1,-2), mul_real+mul_imag).squeeze()
            results.append(result)
        results = torch.stack(results,dim = -1)
        return(results)
        
if __name__ == '__main__':
    model = ComplexMeasurement2(20, units=3)
    a = torch.randn(5,10,20)
    b = torch.randn(5,10,20)
    w = torch.rand(5,10,1)
#   
    y_pred = model([a,b,w])
    print(y_pred)
    kernel_float = model.kernel.detach().numpy()
    kernel_float = kernel_float[:,:,0]+1j*kernel_float[:,:,1]
    weight_float = w.numpy()
    a_float = a.numpy()
    b_float = b.numpy()
    input_float = a_float + 1j*b_float
    if np.ndim(input_float) == 3:
        for aa in range(3):
            for bb in range(5):
                input_f = input_float[bb,:,:]
                kernel_f = kernel_float[aa,:]
                weight_f = weight_float[bb,:]
                input_matrix = np.zeros((20,20),dtype = 'complex64')
                for i in range(10):
                    input_f_H = input_f[i,:].transpose().conjugate()
                    
                    input_matrix = input_matrix + weight_f[i]* np.outer(input_f[i,:],input_f_H)
                    
                kernel_f_H = kernel_f.transpose().conjugate()
                kernel_matrix = np.outer(kernel_f, kernel_f_H)
                
                res = sum(np.diag(np.matmul(input_matrix,kernel_matrix)))
                    
                print(res)
    else:
        for aa in range(3):
            for bb in range(5):
                for t in range(10):
                    input_f = input_float[bb,:,t,:]
                    kernel_f = kernel_float[aa,:]
                    weight_f = weight_float[bb,:,t,:]
                    input_matrix = np.zeros((20,20),dtype = 'complex64')
                    for i in range(2):
                        input_f_H = input_f[i,:].transpose().conjugate()
                        
                        input_matrix = input_matrix + weight_f[i]* np.outer(input_f[i,:],input_f_H)
                        
                    kernel_f_H = kernel_f.transpose().conjugate()
                    kernel_matrix = np.outer(kernel_f, kernel_f_H)
                    
                    res = sum(np.diag(np.matmul(input_matrix,kernel_matrix)))

                    
                    print(np.real(res))

