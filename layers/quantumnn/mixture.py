# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class QMixture(torch.nn.Module):

    def __init__(self, use_weights=True, device = torch.device('cuda')):
        super(QMixture, self).__init__()
        self.use_weights = use_weights
        self.device = device
        
    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
        in_modalities = inputs[0] #[modal_1,...modal_n], each being a list of [real, imag] arrays
        
        weights = inputs[1].transpose(0,1) #(time_stamps, batch_size, num_modalities)
        embed_dim = in_modalities[0][0][0].shape[-1]
        outputs = []
        for reps_t in zip(*in_modalities, weights):
            multimodal_rep = [torch.stack(rep_field, dim = -1) for rep_field in zip(*reps_t[:-1])] 
            w = reps_t[-1].unsqueeze(dim = 1).unsqueeze(dim = -1).expand(-1, embed_dim,-1,-1)
            output_rep = [torch.matmul(_rep, w).squeeze(dim = -1) for _rep in multimodal_rep]
            outputs.append(output_rep)
            
        return outputs
