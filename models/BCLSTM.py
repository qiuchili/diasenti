# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class BCLSTM(nn.Module):
    def __init__(self, opt):
        super(BCLSTM, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims
        self.total_input_dim = sum(self.input_dims)
        
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
          
        if type(opt.fc_dims) == int:
            self.fc_dims = [opt.fc_dims]
        else:
            self.fc_dims = [int(s) for s in opt.fc_dims.split(',')]
        
        self.dialogue_hidden_dim = opt.dialogue_hidden_dim
        self.dialogue_fc_dim = opt.dialogue_fc_dim
        
    
        #if opt.embedding_enabled:
        #    embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        #    self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
            
        self.n_classes = opt.output_dim
        self.output_dropout_rate = opt.output_dropout_rate
        
        self.lstms = nn.ModuleList([nn.LSTMCell(input_dim, hidden_dim) \
                                   for input_dim, hidden_dim in \
                                   zip(self.input_dims, self.hidden_dims)])
    
        self.drop_outs = nn.ModuleList([nn.Dropout(self.output_dropout_rate) \
                                        for i in range(len(self.hidden_dims))]) 
    
    
    
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, fc_dim) \
                                   for hidden_dim, fc_dim in \
                                   zip(self.hidden_dims, self.fc_dims)])
    
    
    
        self.dialogue_lstm = nn.LSTMCell(sum(self.fc_dims),self.dialogue_hidden_dim)
        
        self.drop_out =  nn.Dropout(self.output_dropout_rate)
        
        #self.fc_out = SimpleNet(self.dialogue_hidden_dim, self.output_cell_dim,
        #                        self.output_dropout_rate, self.output_dim)

        self.fc_out = nn.Linear(self.dialogue_hidden_dim, self.dialogue_fc_dim)
        
        self.smax_fc    = nn.Linear(self.dialogue_fc_dim, self.n_classes)
        
        
    
    
    def forward(self, in_modalities):
        umask = in_modalities[-1]
        in_modalities = in_modalities[:-2]
        
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        
        #Unimodal
        all_h = []
        
        for modality, dim, lstm, dropout, fc in zip(in_modalities,self.hidden_dims,self.lstms, self.drop_outs, self.fcs):
           
            self.h = torch.zeros(batch_size, dim).to(self.device)
            self.c = torch.zeros(batch_size, dim).to(self.device)
            
            h = []
            for t in range(time_stamps):
                #Apply the mask dirrectly on the data
                input_u = modality[:,t,:]*umask[:,t].unsqueeze(dim=-1)
                self.h, self.c = lstm(input_u, (self.h,self.c))
                self.h = torch.tanh(self.h)
                self.h = dropout(self.h)
                h.append(torch.tanh(fc(self.h)))
                
            all_h.append(h)
            
        
            
        #Multimodal
        utterance_features = [torch.stack(h, dim = -2) for h in all_h]
        
        dialogue_utterance_feature = torch.cat(utterance_features, dim=-1)
        
    
        self.h_dialogue = torch.zeros(batch_size, self.dialogue_hidden_dim).to(self.device)
        self.c_dialogue = torch.zeros(batch_size, self.dialogue_hidden_dim).to(self.device)
        
        all_h_dialogue = []
        for t in range(time_stamps):
            input_m = dialogue_utterance_feature[:,t,:]*umask[:,t].unsqueeze(dim=-1)
            self.h_dialogue, self.c_dialogue = self.dialogue_lstm(input_m, (self.h_dialogue,self.c_dialogue)) 
            self.h_dialogue = self.drop_out(self.h_dialogue)
            all_h_dialogue.append(torch.tanh(self.fc_out(self.h_dialogue)))
            
        
        output = [self.smax_fc(_h) for _h in all_h_dialogue]
        
        #Stack hidden states
        output = torch.stack(output, dim=-2)
        
        log_prob = F.log_softmax(output, 2) # batch, seq_len,  n_classes
        
            
        return log_prob
