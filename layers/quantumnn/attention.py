# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn
import torch.nn.functional as F

class QAttention(torch.nn.Module):
    def __init__(self):
        super(QAttention, self).__init__()


    def forward(self, in_states, mea_states, mea_weights):
        

        out_states = []
        mea_r = mea_states[0]
        mea_i = mea_states[1]
        mea_mat_r = torch.matmul(mea_r.unsqueeze(dim = -1), mea_r.unsqueeze(dim = -2)) \
        + torch.matmul(mea_i.unsqueeze(dim = -1), mea_i.unsqueeze(dim = -2))
        mea_mat_i = torch.matmul(mea_r.unsqueeze(dim = -1), mea_r.unsqueeze(dim = -2)) \
        + torch.matmul(mea_i.unsqueeze(dim = -1), mea_i.unsqueeze(dim = -2))
        
        time_stamps = mea_r.shape[1]
        for s in in_states:
            s_r = s[0] # Real part
            s_i = s[1] # Imaginary part

            probs = []
            for i in range(time_stamps):
                m_r, m_i = mea_r[:,i,:].unsqueeze(dim = 1), mea_i[:,i,:].unsqueeze(dim = 1)
                prob = self.measurement(m_r, m_i, s_r, s_i)
                probs.append(prob)
                
            weights = torch.cat(probs,dim=-1)* mea_weights.squeeze(dim = -1)
            weights = F.softmax(weights, dim = -1)
            
            out_r = torch.sum(mea_mat_r * weights.unsqueeze(dim = -1).unsqueeze(dim = -1),dim = 1)
            out_i = torch.sum(mea_mat_i * weights.unsqueeze(dim = -1).unsqueeze(dim = -1),dim = 1)
            out_states.append([out_r, out_i])
        return out_states
            
    
    # p = (s_r+i*s_i)(rho_r +i*rho_i)(s_r' - i*s_i')
    def measurement(self, s_r, s_i, rho_r, rho_i):
        res_r = torch.matmul(s_r, rho_r) - torch.matmul(s_i,rho_i)
        res_i = torch.matmul(s_r, rho_i) + torch.matmul(s_i,rho_r)
         
        prob = torch.matmul(res_r, s_r.transpose(1,2)) + torch.matmul(res_i, s_i.transpose(1,2))
        #res_i_2 = - torch.matmul(res_r, rho_i.transpose(1,2)) + torch.matmul(res_i, rho_r.transpose(1,2))
    
        return prob.squeeze(dim = -1)