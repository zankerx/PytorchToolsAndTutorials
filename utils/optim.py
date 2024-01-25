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

    param_groups =  [
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

class StepWarmupCosinScheduler():

    def __init__(self, warmup_steps, total_steps, start_lr, max_lr, final_lr):
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.current_step = 0

    def adjust_lr(self, optimizer, step):

        #compute lr

        #warmup
        if step < self.warmup_steps:
            lr = self.start_lr + (self.max_lr - self.start_lr) * step / self.warmup_steps

        #cosin
        else:
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1.0 + math.cos(math.pi * (self.steps - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        

        for param_group in optimizer.param_groups:    
            param_group["lr"] = lr
