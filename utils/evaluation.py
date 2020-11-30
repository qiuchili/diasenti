# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

def evaluate(params, outputs, targets):

    outputs_max_ids = outputs.argmax(dim = -1).cpu().numpy()
    targets_max_ids = targets.cpu().numpy()
    acc = accuracy_score(outputs_max_ids, targets_max_ids)
    weighted_f1 = f1_score(targets_max_ids,outputs_max_ids,average='weighted')
    performance_dict = {'acc':acc,'f1':weighted_f1}
    
    report = classification_report(targets_max_ids, outputs_max_ids, target_names=params.emotion_dic, output_dict = True)

    for _id, emo in enumerate(params.emotion_dic):
        intersection = sum((outputs_max_ids == _id) * (targets_max_ids == _id))
        union = sum(((outputs_max_ids == _id) + (targets_max_ids == _id))>0)
        acc = intersection/union
        f1 = report[emo]['f1-score']  
        
        #WA = (TP *N/P + TN)/2N
        performance_dict[emo+'_f1'] = f1
        performance_dict[emo+'_precision'] = report[emo]['precision']
        performance_dict[emo+'_recall'] = report[emo]['recall']
        
    return performance_dict