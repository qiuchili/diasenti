# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
    
class CMN(nn.Module):

    def __init__(self, opt):
        super(CMN, self).__init__()

        self.D_m       = sum(opt.input_dims)
        self.D_h       = opt.hidden_dim
        self._time_stamps = opt.context_len
        self._hops = opt.hops
        self.n_classes = opt.output_dim
        self.own_history_gru = nn.GRUCell(self.D_m, self.D_h)
        self.other_history_gru = nn.GRUCell(self.D_m, self.D_h)
        self.device = opt.device
        
        self.local_dropout = nn.Dropout(opt.local_gru_dropout)
        self.memory_rnn_own = nn.GRU(self.D_h, self.D_h)
        self.memory_rnn_other = nn.GRU(self.D_h, self.D_h)
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
        other_hidden = self.local_dropout(other_hidden)


        # Combine the two
        own_hidden = self.activation(own_hidden)
        other_hidden = self.activation(other_hidden)
        all_mask = _histories_own_mask + _histories_other_mask
        input_proj = self.activation(self.input_dense(batch_input)) #(batch_size, hidden_dim)
        
        # Multi-hop Memory Module
        for hop in range(self._hops):
            if hop == 0:
                input_rnn_outputs_own, final_state = self.memory_rnn_own(own_hidden) #(batch_size, time_steps, hidden_dim)
                output_rnn_outputs_own, final_state = self.memory_rnn_own(own_hidden)
            else:
                input_rnn_outputs_own = output_rnn_outputs_own
                output_rnn_outputs_own, final_state = self.memory_rnn_own(own_hidden)
                
                
            output_rnn_outputs_own = output_rnn_outputs_own * _histories_own_mask.unsqueeze(dim=-1).expand_as(output_rnn_outputs_own) #(batch_size, time_steps, hidden_dim)
                
            #Attentional Read Operation from rnn_output memories
            attScore = self.activation(torch.squeeze(torch.bmm(input_rnn_outputs_own, input_proj.unsqueeze(dim=-1)))) #(batch_size, time_steps)
            attScore = all_mask* attScore + (1-all_mask)* torch.ones_like(attScore)*(-10000)
            attScore = F.softmax(attScore, dim = -1) #(batch_size, time_steps)
            attScore = self.attention_dropout(attScore) # (batch_size, time_steps)
            attScore = all_mask* attScore
            weighted_own= torch.squeeze(torch.bmm(attScore.unsqueeze(dim=-2), output_rnn_outputs_own)) #(batch_size, hidden_dim) 
            
            if hop == 0:
                input_rnn_outputs_other, final_state = self.memory_rnn_other(other_hidden)
                output_rnn_outputs_other, final_state = self.memory_rnn_other(other_hidden)
            else:
                input_rnn_outputs_other = output_rnn_outputs_other
                output_rnn_outputs_other, final_state = self.memory_rnn_other(other_hidden)
                    
            output_rnn_outputs_other = output_rnn_outputs_other * _histories_other_mask.unsqueeze(dim=-1).expand_as(output_rnn_outputs_other) #(batch_size, time_steps, hidden_dim)
                
            #Attentional Read Operation from rnn_output memories
            attScore = self.activation(torch.squeeze(torch.bmm(input_rnn_outputs_other, input_proj.unsqueeze(dim=-1)))) #(batch_size, time_steps)
            attScore = all_mask* attScore + (1-all_mask)* torch.ones_like(attScore)*(-10000)
            attScore = F.softmax(attScore, dim = -1) #(batch_size, time_steps)
            attScore = self.attention_dropout(attScore) # (batch_size, time_steps)
            attScore = all_mask* attScore
            weighted_other= torch.squeeze(torch.bmm(attScore.unsqueeze(dim=-2), output_rnn_outputs_other)) #(batch_size, hidden_dim) 
            
            input_proj = self.activation(input_proj + weighted_own + weighted_other)
                    
        output = F.log_softmax(self.output_dense(input_proj), dim = -1)
        return output



