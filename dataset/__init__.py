# -*- coding: utf-8 -*-
def setup(opt):
    if opt.dataset_name.lower() == 'iemocap':
        from dataset.iemocap_reader import IEMOCAPReader as MMDataReader
    elif opt.dataset_name.lower() == 'meld':    
        from dataset.meld_reader import MELDReader as MMDataReader
    elif 'avec' in opt.dataset_name.lower():    
        from dataset.avec_reader import AVECReader as MMDataReader
    else:
        #Default
        from dataset.meld_reader import MELDReader as MMDataReader

    reader = MMDataReader(opt)
    return reader