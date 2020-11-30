# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np

def PhaseEmbedding(input_dim, embedding_dim, embedding_matrix = None, sentiment_dic = None, freeze = False):
    phase_embedding_matrix = torch.empty(input_dim,embedding_dim)
#    nn.init.uniform_(phase_embedding_matrix,0, 2*np.pi)
    if not embedding_matrix is None:
        phase_embedding_matrix = torch.tensor(embedding_matrix, requires_grad = True)
    else:
        nn.init.zeros_(phase_embedding_matrix)
        if not sentiment_dic is None:
            pos_ind = len([ _i  for _i,x in enumerate(sentiment_dic) if x==-1.0])
            neg_ind = len([ _i  for _i,x in enumerate(sentiment_dic) if x==-1.0])
            neu_ind = len([ _i  for _i,x in enumerate(sentiment_dic) if x==1.0])
            phase_embedding_matrix[pos_ind,:] = 0
            phase_embedding_matrix[neg_ind,:] = np.pi
            phase_embedding_matrix[neu_ind,:] = 0.5*np.pi
    
    embedding_layer = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
    return embedding_layer

def WeightEmbedding(input_dim, freeze = False):   
    weight_embedding_matrix = torch.empty(input_dim,1)
    nn.init.constant_(weight_embedding_matrix, 1)
    embedding_layer = nn.Embedding.from_pretrained(weight_embedding_matrix, freeze=freeze)
    return embedding_layer

  
class ComplexEmbedding(torch.nn.Module):
    def __init__(self, embedding_matrix, amplitude_freeze=False, phase_freeze=False):
        super(ComplexEmbedding, self).__init__()

        amplitude_embedding_matrix = torch.abs(embedding_matrix)
        self.amplitude_embed = nn.Embedding.from_pretrained(amplitude_embedding_matrix, freeze=amplitude_freeze)
        phase_embedding_matrix = torch.empty_like(amplitude_embedding_matrix)
        nn.init.uniform_(phase_embedding_matrix,0, 2*np.pi)      
        phase_embedding_matrix = phase_embedding_matrix + math.pi * (1 - torch.sign(embedding_matrix)) / 2 # based on [0, 2*pi]
        self.phase_embed = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=phase_freeze)
        
    def forward(self, indices):
        amplitude_embed = self.amplitude_embed(indices)
        phase_embed = self.phase_embed(indices)

        return [amplitude_embed, phase_embed]
