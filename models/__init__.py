from .BCLSTM import BCLSTM
from .DialogueRNN import BiDialogueRNN
from .DialogueGCN import DialogueGCN
from .ICON import ICON
from .CMN import CMN
from .QAttN import QAttN
from .QMN import QMN
from .QMNAblation import QMNAblation
from .CCMF import CCMF
from .QMPN import QMPN
from .EFLSTM import EFLSTM
from .TFN import TFN 
from .MARN import MARN
from .RMFN import RMFN
from .LMF import LMF
from .MFN import MFN
from .LSTHM import LSTHM
from .LFLSTM import LFLSTM
from .MULT import MULT

def setup(opt):
    
    print("network type: " + opt.network_type)
    if opt.network_type == "ef-lstm":
        model = EFLSTM(opt)
    elif opt.network_type == "tfn":
        model = TFN(opt)
    elif opt.network_type == "marn":
        model = MARN(opt)
    elif opt.network_type == "rmfn":
        model = RMFN(opt)
    elif opt.network_type == 'lmf':
        model = LMF(opt)
    elif opt.network_type == 'mfn':
        model = MFN(opt)
    elif opt.network_type == 'lsthm':
        model = LSTHM(opt)
    elif opt.network_type == 'lf-lstm':
        model = LFLSTM(opt)
    elif opt.network_type == 'multimodal-transformer':
        model = MULT(opt)
    elif opt.network_type == 'bc-lstm':
        model = BCLSTM(opt)
    elif opt.network_type == 'dialogue-rnn':
        model = BiDialogueRNN(opt)
    elif opt.network_type == 'dialogue-gcn':
        model = DialogueGCN(opt)
    elif opt.network_type == 'icon':
        model = ICON(opt)
    elif opt.network_type == 'cmn':
        model = CMN(opt)
    elif opt.network_type == 'qattn':
        model = QAttN(opt)
    elif opt.network_type == 'qmn':
        model = QMN(opt)
    elif opt.network_type == 'qmn-ablation':
        model = QMNAblation(opt)
    elif opt.network_type == 'ccmf':
        model = CCMF(opt)
    elif opt.network_type == 'qmpn':
        model = QMPN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
