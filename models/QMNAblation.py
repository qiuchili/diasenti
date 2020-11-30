# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers.quantumnn.embedding import PositionEmbedding
from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.mixture import QMixture
from layers.quantumnn.rnn import QRNNCell
from layers.quantumnn.measurement import QMeasurement
from layers.complexnn.measurement import ComplexMeasurement
from layers.quantumnn.outer import QOuter
from models.SimpleNet import SimpleNet
from layers.complexnn.l2_norm import L2Norm
from layers.quantumnn.dense import QDense
from layers.quantumnn.dropout import QDropout
from layers.quantumnn.activation import QActivation
from layers.quantumnn.activation2 import QActivation2


class QMNAblation(nn.Module):
    def __init__(self, opt):
        super(QMNAblation, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.speaker_num = opt.speaker_num
        self.dataset_name = opt.dataset_name
        
        
        # MELD data 
        if self.dataset_name.lower() == 'meld':
            self.speaker_num = 1
        self.n_classes = opt.output_dim
        self.input_concat = opt.input_concat
        self.zero_phase = opt.zero_phase
        self.measurement_type = opt.measurement_type
        self.classical_recurrent = opt.classical_recurrent
        self.quantum_recurrent = opt.quantum_recurrent

        
        if self.input_concat:
            self.projections = nn.ModuleList([nn.Linear(self.total_input_dim, self.embed_dim)])
        
        elif self.classical_recurrent:
            self.projections = nn.ModuleList([nn.GRU(dim, self.embed_dim, 1) for dim in self.input_dims])
        else:
            self.projections = nn.ModuleList([nn.Linear(dim, self.embed_dim) for dim in self.input_dims])

        
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.norm = L2Norm(dim = -1)
        self.mixture = QMixture(device = self.device)
        self.output_cell_dim = opt.output_cell_dim
        self.phase_embeddings = nn.ModuleList([PositionEmbedding(self.embed_dim, input_dim = self.speaker_num,\
                                                                 zero_phase = self.zero_phase, device = self.device)]*len(self.input_dims)) 
        
        self.out_dropout_rate = opt.out_dropout_rate
        self.num_layers = opt.num_layers
        self.recurrent_cells = nn.ModuleList([QRNNCell(self.embed_dim, device = self.device)]*self.num_layers)
        self.out_dropout = QDropout(p=self.out_dropout_rate)
        self.activation = QActivation(scale_factor = 1,beta = 0.8)
        
        self.measurement_type = opt.measurement_type
        if self.measurement_type == 'quantum':
            self.measurement = QMeasurement(self.embed_dim)
            self.fc_out = SimpleNet(self.embed_dim, self.output_cell_dim,
                                    self.out_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
        
        elif self.measurement_type == 'flatten':
            self.fc_out = SimpleNet(self.embed_dim, self.output_cell_dim,
                                    self.out_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
        else:
            self.measurement = ComplexMeasurement(self.embed_dim, units = self.embed_dim)
            self.fc_out = SimpleNet(self.embed_dim, self.output_cell_dim,
                                    self.out_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
            
        
    def get_params(self):
    
        unitary_params = []
        remaining_params = []
        for i in range(self.num_layers):
            unitary_params.append(self.recurrent_cells[i].unitary_x)
            unitary_params.append(self.recurrent_cells[i].unitary_h)
            remaining_params.append(self.recurrent_cells[i].Lambda)
        
        remaining_params.extend(list(self.projections.parameters()))

        remaining_params.extend(list(self.phase_embeddings.parameters()))
        for i in range(self.num_layers):
            remaining_params.append(self.recurrent_cells[i].Lambda)
            
        if self.measurement_type == 'quantum':
            unitary_params.extend(list(self.measurement.parameters()))
        else:
            remaining_params.extend(list(self.measurement.parameters()))
        remaining_params.extend(list(self.fc_out.parameters()))
            
        return unitary_params, remaining_params
    
    def forward(self, in_modalities):
        smask = in_modalities[-2] # Speaker ids
        in_modalities = in_modalities[:-2]
        
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        
        # Project All modalities of each utterance to the same space
        #utterance_reps = [nn.Tanh()(projection(x)) for x, projection in zip(in_modalities,self.projections)] 
        if self.input_concat:
            in_modalities = torch.cat(in_modalities, dim=-1)
        
        
        if self.classical_recurrent:
            utterance_reps = [nn.ReLU()(projection(x)[0]) for x, projection in zip(in_modalities,self.projections)] 
        else:
            utterance_reps = [nn.ReLU()(projection(x)) for x, projection in zip(in_modalities,self.projections)] 

        # Take the amplitudes 
        # multiply with modality specific vectors to construct weights
        weights = [self.norm(rep) for rep in utterance_reps]
        weights = F.softmax(torch.cat(weights, dim = -1), dim = -1)
        
        amplitudes = [F.normalize(rep, dim = -1) for rep in utterance_reps]
        phases = [phase_embed(smask.argmax(dim = -1)) for phase_embed in self.phase_embeddings]
        unimodal_pure = [self.multiply([phase, amplitude]) for phase, amplitude in zip(phases,amplitudes)]
        unimodal_matrices = [self.outer(s) for s in unimodal_pure]
        in_states = self.mixture([unimodal_matrices, weights])
        
        if self.quantum_recurrent:
            for l in range(self.num_layers):
                # Initialize the cell h
                h_r = torch.stack(batch_size*[torch.eye(self.embed_dim)/self.embed_dim],dim =0)
                h_i = torch.zeros_like(h_r)
                h = [h_r.to(self.device),h_i.to(self.device)]
                all_h = []
                for t in range(time_stamps):
                    h = self.recurrent_cells[l](in_states[t],h)
                    all_h.append(self.activation(h))
                in_states = all_h

        output = []
        
        for _h in in_states:
#            _h = self.out_dropout(_h)
#            _h = self.dense(_h)
#            measurement_probs = self.measurement(_h)
            if self.measurement_type == 'flatten':
                _output = self.fc_out(_h[0].reshape(batch_size,-1))
            
            else:
                measurement_probs = self.measurement(_h)
                _output = self.fc_out(measurement_probs)
            output.append(_output)
            
            
        output = torch.stack(output, dim=-2)
        log_prob = F.log_softmax(output, 2) # batch, seq_len,  n_classes

        return log_prob
