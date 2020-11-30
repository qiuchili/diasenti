# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
    
class ICON(nn.Module):

    def __init__(self, opt):
        super(ICON, self).__init__()

        self.D_m       = sum(opt.input_dims)
        self.D_h       = opt.hidden_dim
        self._time_stamps = opt.context_len
        self._hops = opt.hops
        self.n_classes = opt.output_dim
        self.own_history_gru = nn.GRUCell(self.D_m, self.D_h)
        self.other_history_gru = nn.GRUCell(self.D_m, self.D_h)
        self.device = opt.device
        
        self.local_dropout = nn.Dropout(opt.local_gru_dropout)
        self.global_rnn = nn.GRU(self.D_h, self.D_h)
        self.memory_rnn = nn.GRU(self.D_h, self.D_h)
        self.input_dense = nn.Linear(self.D_m, self.D_h,bias = False)
        self.activation = nn.Tanh()
        self.attention_dropout = nn.Dropout(opt.local_gru_dropout)
        self.output_dense = nn.Linear(self.D_h, self.n_classes, bias = True)
#         q = tf.contrib.layers.fully_connected(
#                    queries,
#                    self._embedding_size,
#                    activation_fn=tf.nn.tanh,
#                    normalizer_fn=None,
#                    normalizer_params=None,
#                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1227),
#                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.001),
#                    biases_initializer=tf.zeros_initializer(),
#                    trainable=True,
#                    scope="input"
#                )


    def forward(self, in_modalities):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        _histories_own_mask, _histories_other_mask = in_modalities[-2], in_modalities[-1]
         
        num_modalities = int((len(in_modalities)-2)/3)
        batch_input = torch.cat(in_modalities[:num_modalities],dim=-1)
        _histories_own = torch.cat(in_modalities[num_modalities:2*num_modalities],dim=-1)
        _histories_other = torch.cat(in_modalities[2*num_modalities:3*num_modalities],dim=-1)
        batch_size = len(batch_input)
        
        ##Input GRU Own
        h = torch.zeros(batch_size, self.D_h).to(self.device)
        
        all_h = []
        for t in range(self._time_stamps):
            new_h = self.own_history_gru(_histories_own[:,t,:], h) #(batch_size, hidden_dim)
            mask_t = _histories_own_mask[:,t]
            expanded_mask = mask_t.unsqueeze(dim=-1).expand_as(new_h)
            h = new_h*expanded_mask + h*(1-expanded_mask)
            all_h.append(h)
            
        own_hidden = torch.stack(all_h, dim = 1) #(batch_size, time_steps, hidden_dim)
        own_hidden = own_hidden * _histories_own_mask.unsqueeze(dim=-1).expand_as(own_hidden)
        own_hidden = self.local_dropout(own_hidden)
        
        ##Input GRU Other
        h = torch.zeros(batch_size, self.D_h).to(self.device)
        
        all_h = []
        for t in range(self._time_stamps):
            new_h = self.other_history_gru(_histories_other[:,t,:], h) #(batch_size, hidden_dim)
            mask_t = _histories_other_mask[:,t]
            expanded_mask = mask_t.unsqueeze(dim=-1).expand_as(new_h)
            h = new_h*expanded_mask + h*(1-expanded_mask)
            all_h.append(h)
            
        other_hidden = torch.stack(all_h, dim = 1) #(batch_size, time_steps, hidden_dim)
        other_hidden = other_hidden * _histories_other_mask.unsqueeze(dim=-1).expand_as(other_hidden)
        other_hidden = self.local_dropout(other_hidden)

        # Combine the two
        all_history = self.activation(own_hidden + other_hidden)
        all_mask = _histories_own_mask + _histories_other_mask
        input_proj = self.activation(self.input_dense(batch_input)) #(batch_size, hidden_dim)
        
        # Multi-hop Memory Module
        for hop in range(self._hops):
            if hop == 0:
                rnn_input = all_history
                rnn_model = self.global_rnn
            else:
                rnn_input = rnn_outputs
                rnn_model = self.memory_rnn
                
            rnn_outputs, final_state = rnn_model(rnn_input)
            rnn_outputs = rnn_outputs * all_mask.unsqueeze(dim=-1).expand_as(rnn_outputs) #(batch_size, time_steps, hidden_dim)
                
            #Attentional Read Operation from rnn_output memories
            attScore = self.activation(torch.squeeze(torch.bmm(rnn_outputs, input_proj.unsqueeze(dim=-1)))) #(batch_size, time_steps)
            attScore = all_mask* attScore + (1-all_mask)* torch.ones_like(attScore)*(-10000)
            attScore = F.softmax(attScore, dim = -1) #(batch_size, time_steps)
            attScore = self.attention_dropout(attScore) #(batch_size, time_steps)
            attScore = all_mask* attScore
            weighted = torch.squeeze(torch.bmm(attScore.unsqueeze(dim=-2), rnn_outputs)) #(batch_size, hidden_dim) 
            input_proj = self.activation(input_proj + weighted)
        
        output = F.log_softmax(self.output_dense(input_proj), dim = -1)
        return output



