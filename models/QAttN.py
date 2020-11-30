# -*- coding: utf-8 -*-
import torch
import time
from torch import nn
import torch.nn.functional as F
from layers.quantumnn.embedding import PositionEmbedding

from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.rnn import QRNNCell
from layers.quantumnn.mixture import QMixture
from layers.quantumnn.measurement import QMeasurement
from layers.complexnn.measurement import ComplexMeasurement
from layers.quantumnn.outer import QOuter
from models.SimpleNet import SimpleNet
from layers.quantumnn.attention import QAttention
from layers.complexnn.l2_norm import L2Norm

class QAttN(nn.Module):
    def __init__(self, opt):
        super(QAttN, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.speaker_num = opt.speaker_num
        self.n_classes = opt.output_dim
        self.output_cell_dim = opt.output_cell_dim
        self.dataset_name = opt.dataset_name

        self.projections = nn.ModuleList([nn.Linear(dim, self.embed_dim) for dim in self.input_dims])
        if self.dataset_name.lower() == 'meld':
            self.speaker_num = 1
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.norm = L2Norm(dim = -1)
        self.mixture = QMixture(device = self.device)
        self.phase_embeddings = nn.ModuleList([PositionEmbedding(self.embed_dim, input_dim = 1, device = self.device)]* len(self.input_dims)) 
        self.out_dropout_rate = opt.out_dropout_rate
        self.attention = QAttention()
        self.num_modalities = len(self.input_dims)
        #self.out_dropout = QDropout(p=self.out_dropout_rate)
        
        #self.dense = QDense(self.embed_dim, self.n_classes)
        self.measurement_type = opt.measurement_type
        if opt.measurement_type == 'quantum':
            self.measurement = QMeasurement(self.embed_dim)
            self.fc_out = SimpleNet(self.embed_dim* self.num_modalities, self.output_cell_dim,
                                    self.out_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
        
        else:
            self.measurement = ComplexMeasurement(self.embed_dim * self.num_modalities, units = 20)
            self.fc_out = SimpleNet(20, self.output_cell_dim,
                                    self.out_dropout_rate,self.n_classes,
                                    output_activation = nn.Tanh())
                
    def get_params(self):
        unitary_params = []
        remaining_params = []
        
        remaining_params.extend(list(self.projections.parameters()))
        remaining_params.extend(list(self.phase_embeddings.parameters()))
            
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
        
        utterance_reps = [nn.ReLU()(projection(x)) for x, projection in zip(in_modalities,self.projections)] 

        # Take the amplitudes 
        # multiply with modality specific vectors to construct weights
        weights = [self.norm(rep) for rep in utterance_reps]

        amplitudes = [F.normalize(rep, dim = -1) for rep in utterance_reps]
        
        phases = [phase_embed(smask.argmax(dim = -1)) for phase_embed in self.phase_embeddings]
        unimodal_pure = [self.multiply([phase, amplitude]) for phase, amplitude in zip(phases,amplitudes)]
        unimodal_matrices = [self.outer(s) for s in unimodal_pure]
        
        
        probs = []
        
        # For each modality
        # we mix the remaining modalities as queries (to_be_measured systems) 
        # and treat the modality features as keys (measurement operators)
        
        for ind in range(self.num_modalities):   
            
            # Obtain mixed states for the rest modalities
            other_weights = [weights[i] for i in range(self.num_modalities) if not i == ind]
            mixture_weights = F.softmax(torch.cat(other_weights, dim = -1), dim = -1)
            other_states = [unimodal_matrices[i] for i in range(self.num_modalities) if not i == ind]     
            q_states = self.mixture([other_states, mixture_weights])
            
            # Obtain pure states and weights for the modality of interest
            k_weights = weights[ind]
            k_states = unimodal_pure[ind]
            
            # Compute cross-modal interactions, output being a list of post-measurement states
            in_states = self.attention(q_states, k_states, k_weights)
            
            # Apply measurement to each output state
            output = []
            for _h in in_states:
                measurement_probs = self.measurement(_h)
                output.append(measurement_probs)

            probs.append(output)
        
        # Concatenate the measurement probabilities per-time-stamp
        concat_probs = [self.fc_out(torch.cat(output_t, dim = -1)) for output_t in zip(*probs)] 
        concat_probs = torch.stack(concat_probs, dim=-2)
        
        log_prob = F.log_softmax(concat_probs, 2) # batch, seq_len,  n_classes


        return log_prob

     