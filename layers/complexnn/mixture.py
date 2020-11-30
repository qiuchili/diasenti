# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexMixture(torch.nn.Module):

    def __init__(self, use_weights=True, device = torch.device('cuda')):
        super(ComplexMixture, self).__init__()
        self.use_weights = use_weights
        self.device = device
        
    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

#        input_real = torch.unsqueeze(inputs[0], dim=-1) 
#        input_imag = torch.unsqueeze(inputs[1], dim=-1) 
        
#        if inputs[2].dim() == inputs[1].dim()-1:
#            weight = torch.unsqueeze(torch.unsqueeze(inputs[2], dim=-1), dim=-1)
#        else:
#            weight = torch.unsqueeze(inputs[2], dim=-1)
        
        weight = inputs[2]
        output_r = torch.zeros(inputs[0].shape[0], inputs[0].shape[-1],inputs[0].shape[-1]).to(self.device)
        output_i = torch.zeros(inputs[0].shape[0], inputs[0].shape[-1],inputs[0].shape[-1]).to(self.device)
        
        for i in range(inputs[0].shape[0]):
            
#        for i in range(len(weight)):
            real_i = inputs[0][i,:,:].to(self.device)
            imag_i = inputs[1][i,:,:].to(self.device)
            weight_i = weight[i,:].to(self.device)
            
            real_o = torch.matmul(torch.unsqueeze(real_i,dim=-1),torch.unsqueeze(real_i,dim=-2)) \
            + torch.matmul(torch.unsqueeze(imag_i,dim=-1),torch.unsqueeze(imag_i,dim=-2))
            
            
            imag_o = torch.matmul(torch.unsqueeze(imag_i,dim=-1),torch.unsqueeze(real_i,dim=-2)) \
            - torch.matmul(torch.unsqueeze(real_i,dim=-1),torch.unsqueeze(imag_i,dim=-2))
            
#            for j in range(inputs[0].shape[0]):
#                real_o[i] = real_o[i] * weight_i[j]
#                imag_o[i] = imag_o[i] * weight_i[j]
            output_r[i] = torch.matmul(real_o.permute(1,2,0),weight_i).squeeze()
            output_i[i] = torch.matmul(imag_o.permute(1,2,0),weight_i).squeeze()
            #         if not self.use_weights:
#            output_r = torch.mean(output_real, dim=1)
#            output_i = torch.mean(output_imag, dim=1)
        
        
#        else:
            
                
            
            
        
#        input_real_transpose = torch.unsqueeze(inputs[0], dim=-2) 
#        input_imag_transpose = torch.unsqueeze(inputs[1], dim=-2) 
#
#
#        # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
#        
#        
#        output_real = torch.matmul(input_real, input_real_transpose) + torch.matmul(input_imag, input_imag_transpose) #shape: (None, 60, 300, 300)
#        output_imag = torch.matmul(input_imag, input_real_transpose) - torch.matmul(input_real, input_imag_transpose) #shape: (None, 60, 300, 300)
#        
#        embed_dim = output_real.shape[-1]
#       
#            
#            output_real = output_real.float() * weight.float()
#            output_r = torch.sum(output_real, dim=1)  #shape: (None, 300, 300)
#            output_imag = output_imag.float() * weight.float()
#            output_i = torch.sum(output_imag, dim=1)  #shape: (None, 300, 300)
        return [output_r, output_i]

def test():
    mixture = ComplexMixture()
    a = torch.randn(3, 4, 10)
    b = torch.randn(3, 4, 10)
    c = torch.randn(3, 4)
    mix = mixture([a, b, c])
    print(mix)
    if mix[0].dim() == 3:
        print('ComplexMixture Test Passed.')
    else:
        print('ComplexMixture Test Failed.')

if __name__ == '__main__':
    test()
