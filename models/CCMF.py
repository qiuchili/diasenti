import torch
from torch import nn
import torch.nn.functional as F
from layers.quantumnn.embedding import *
from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.outer import QOuter
from layers.complexnn.measurement import ComplexMeasurement
from models.SimpleNet import SimpleNet
from layers.quantumnn.mixture import QMixture


class CCMF(nn.Module):
    
    def __init__(self, opt):
        super(CCMF, self).__init__()
        
        self.device = opt.device    
        self.input_dims = opt.input_dims
        self.speaker_num = opt.speaker_num
        self.embed_dim = opt.embed_dim
        self.num_measurements = opt.num_measurements
        self.output_cell_dim = opt.output_cell_dim
        self.n_classes = opt.output_dim
        
        #if type(opt.hidden_dims) == int:
        #    self.hidden_dims = [opt.hidden_dims]
        #else:
        #    self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
        
        self.output_dropout_rate = opt.output_dropout_rate
        
        
        self.lstms = nn.ModuleList([nn.LSTMCell(input_dim, self.embed_dim) \
                                   for input_dim in self.input_dims])
    
        self.drop_outs = nn.ModuleList([nn.Dropout(self.output_dropout_rate)]* len(self.input_dims))
        
        
        #self.fcs = nn.ModuleList([nn.Linear(hidden_dim, self.embed_dim) \
        #                           for hidden_dim in self.hidden_dims])
        
        self.phase_embeddings = nn.ModuleList([PhaseEmbedding(self.speaker_num, self.embed_dim)]* len(self.input_dims)) 
        
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.measurement = ComplexMeasurement(self.embed_dim,units = self.num_measurements)
        
        
        self.fc_out = SimpleNet(self.num_measurements* len(self.input_dims), self.output_cell_dim,
                                    self.output_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
        
    def forward(self, in_modalities):
        
        umask = in_modalities[-1] # Utterance Mask
        smask = in_modalities[-2] # Speaker ids
        
        in_modalities = in_modalities[:-2]
        
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1] 
        
        #Unimodal
        all_h = []
        
        for modality, lstm, drop_out  in zip(in_modalities,self.lstms,self.drop_outs):
            self.h = torch.zeros(batch_size, self.embed_dim).to(self.device)
            self.c = torch.zeros(batch_size, self.embed_dim).to(self.device)
            
            h = []
            for t in range(time_stamps):
                input_u = modality[:,t,:]*umask[:,t].unsqueeze(dim=-1)
                self.h, self.c = lstm(input_u, (self.h,self.c))
                self.h = torch.tanh(self.h)
                self.h = drop_out(self.h)
                h.append(self.h)
                
            all_h.append(h)
            
        #Amplitudes and Phases for each modality
        utterances = [torch.stack(h, dim = -2) for h in all_h]  
        amplitudes = [F.normalize(utter, dim=-1) for utter in utterances]                
        phases = [phase_embed(smask.argmax(dim = -1)) for phase_embed in self.phase_embeddings]
        
        #Returns Re and Im parts for each modality
        unimodal_pure = [self.multiply([phase, amplitude]) for phase, amplitude in zip(phases,amplitudes)]
        unimodal_density = [self.outer(s) for s in unimodal_pure]
        
        
        unimodal_embeddings = []
        for i in range(len(self.input_dims)):
            output = []
            for t in range(time_stamps):
                #_output =  self.measurement([unimodal_density[i][0][:,t,:,:],unimodal_density[i][1][:,t,:,:]])
                _output =  self.measurement(unimodal_density[i][t])
                output.append(_output)
            output = torch.stack(output, dim=-2)
            unimodal_embeddings.append(output)
         
            
        multimodal_embedding = torch.cat([unimodal_embeddings[0],unimodal_embeddings[1]], dim=-1)     
        pred = self.fc_out(multimodal_embedding) 
        log_prob = F.log_softmax(pred, 2)   # batch, seq_len,  n_classes
        
        return log_prob
        
