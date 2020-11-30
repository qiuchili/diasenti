# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import copy
from utils.evaluation import evaluate
import time
import pickle
from optimizer import RMSprop_Unitary
from models.DialogueRNN import MaskedNLLLoss
def train(params, model):
    criterion = get_criterion(params)
    #unitary_parameters = get_unitary_parameters(model)
    if hasattr(model,'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []
        
    if len(unitary_params)>0:
        unitary_optimizer = RMSprop_Unitary(unitary_params,lr = params.unitary_lr)

    #remaining_parameters = get_remaining_parameters(model,unitary_parameters)
    optimizer = torch.optim.RMSprop(remaining_params,lr = params.lr)  
    
    # Temp file for storing the best model 
    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp',temp_file_name)

    best_val_loss = 99999.0
#    best_val_loss = -1.0
    for i in range(params.epochs):
        print('epoch: ', i)
        model.train()
        with tqdm(total = params.train_sample_num) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(params.reader.get_data(iterable = True, shuffle = True, split='train'),0):
#                For debugging, please run the line below
#                _i,data = next(iter(enumerate(params.reader.get_data(iterable = True, shuffle = True, split = 'train'),0)))

                b_inputs = [inp.to(params.device) for inp in data[:-1]]
                b_targets = data[-1].to(params.device)
                
                # Does not train if batch_size is 1, because batch normalization will crash
                if b_inputs[0].shape[0] == 1:
                    continue
                
                optimizer.zero_grad()
                if len(unitary_params)>0:
                    unitary_optimizer.zero_grad()

                outputs = model(b_inputs)
                b_targets, outputs, loss = get_loss(params, criterion, outputs, b_targets, b_inputs[-1])
                if np.isnan(loss.item()):
                    torch.save(model,params.best_model_file)
                    raise Exception('loss value overflow!')
                    #break           
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), params.clip)
                optimizer.step()
                if len(unitary_params)>0:
                    unitary_optimizer.step()
                    
                # Compute Training Accuracy                                  
                n_total = len(outputs)   
                n_correct = (outputs.argmax(dim = -1) == b_targets).sum().item()
                train_acc = n_correct/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
        
        model.eval()
        
        #################### Compute Validation Performance##################
        val_output,val_target, val_mask = get_predictions(model, params, split = 'dev')
             
        val_target, val_output, val_loss = get_loss(params, criterion,val_output,val_target, val_mask)

                
        print('validation performance:')
        performances = evaluate(params,val_output,val_target)        
        
        print('val_acc = {}, val_loss = {}'.format(performances['acc'], val_loss))
        ##################################################################
        if val_loss < best_val_loss:
            torch.save(model,params.best_model_file)
            print('The best model up till now. Saved to File.')
            best_val_loss = val_loss
        
def get_criterion(params):
    # For ICON, CMN, NLLLoss is used
    if params.dialogue_context:    
        criterion = nn.NLLLoss()
    # For BC-LSTM, DialogueRNN and DialogueGCN, MaskedNLLLoss is used
    else:
        criterion = MaskedNLLLoss(params.loss_weights.to(params.device))   
    return criterion

def get_loss(params, criterion, outputs, b_targets, mask):
    
    # b_targets: (batch_size, dialogue_length, output_dim)
    # outputs: (batch_size, dialogue_length, output_dim)    
    b_targets = b_targets.reshape(-1, params.output_dim).argmax(dim=-1)
    outputs = outputs.reshape(-1, params.output_dim)
    
    if params.dialogue_context:
        loss = criterion(outputs,b_targets)
    else:
        loss = criterion(outputs,b_targets,mask)
        nonzero_idx = mask.view(-1).nonzero()[:,0]
        outputs = outputs[nonzero_idx]
        b_targets = b_targets[nonzero_idx]
            
    return b_targets, outputs, loss

def test(model,params):
    model.eval()
    test_output,test_target, test_mask = get_predictions(model, params, split = 'test')    

    test_target = torch.argmax(test_target.reshape(-1, params.output_dim),-1)
    test_output = test_output.reshape(-1, params.output_dim)
    if not params.dialogue_context:
        nonzero_idx = test_mask.view(-1).nonzero()[:,0]
        test_output = test_output[nonzero_idx]
        test_target = test_target[nonzero_idx]
            
    performances = evaluate(params,test_output,test_target)
    
    return performances

def print_performance(performance_dict, params):
    performance_str = ''
    for key, value in performance_dict.items():
        performance_str = performance_str+ '{} = {} '.format(key,value)
    print(performance_str)
    return performance_str

def get_predictions(model, params, split ='dev'):
    outputs = []
    targets = []
    masks = []
    iterator = params.reader.get_data(iterable =True, shuffle = False, split = split)
        
    for _ii,data in enumerate(iterator,0):  
        data_x = [inp.to(params.device) for inp in data[:-1]]
        data_t = data[-1].to(params.device)
        data_o = model(data_x)
        if not params.dialogue_context:
            masks.append(data_x[-1])
                        
        outputs.append(data_o.detach())
        targets.append(data_t.detach())
            
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    if not params.dialogue_context:   
        masks = torch.cat(masks)
        
    return outputs, targets, masks

def save_model(model,params,s):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    params.dir_name = str(round(time.time()))
    dir_path = os.path.join('tmp',params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    torch.save(model.state_dict(),os.path.join(dir_path,'model'))
#    copyfile(params.config_file, os.path.join(dir_path,'config.ini'))
    params.export_to_config(os.path.join(dir_path,'config.ini'))
    params_2 = copy.deepcopy(params)
    if 'lookup_table' in params_2.__dict__:
        del params_2.lookup_table
    if 'sentiment_dic' in params_2.__dict__:
        del params_2.sentiment_dic
    del params_2.reader
    pickle.dump(params_2, open(os.path.join(dir_path,'config.pkl'),'wb'))
    
    del params_2
    if 'save_phases' in params.__dict__ and params.save_phases:
        print('Saving Phases.')
        phase_dict = model.get_phases()
        for key in phase_dict:
            file_path = os.path.join(dir_path,'{}_phases.pkl'.format(key))
            pickle.dump(phase_dict[key],open(file_path,'wb'))
    eval_path = os.path.join(dir_path,'eval')
    with open(eval_path,'w') as f:
        f.write(s)
    
def save_performance(params, performance_dict):
    df = pd.DataFrame()
    output_dic = {'dataset' : params.dataset_name,
                    'modality' : params.features,
                    'network' : params.network_type,
                    'model_dir_name': params.dir_name}
    output_dic.update(performance_dict)
    df = df.append(output_dic, ignore_index = True)

    if not 'output_file' in params.__dict__:
        params.output_file = 'eval/{}_{}.csv'.format(params.dataset_name, params.network_type)
    df.to_csv(params.output_file, encoding='utf-8', index=True)
