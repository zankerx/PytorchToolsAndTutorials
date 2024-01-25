import torch
import math


'''
Optim parameters
No weight decay on 1d params
Scheduler
'''

def get_params_nowd1d(model):
    
    # no wd on bias/1D params
    # 1 groups : wd, no wd
    # if token in parameter name : no wd.

    param_groups = [
        {
            'params': (p for n, p in model.named_parameters() if (('bias' not in n) and ('token' not in n)) and (len(p.size()) != 1)),
            'has_weight_decay': True, #Scheduler flag
            
        }, 
        {
            'params': (p for n, p in model.named_parameters() if (('bias' in n) or ('token' in n)) or (len(p.size()) == 1)),
            'has_weight_decay': False, #Scheduler flag
            'weight_decay': 0.0
        }
    ]

    return param_groups