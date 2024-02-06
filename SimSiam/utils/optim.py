import torch
import math



def adjust_lr(optimizer, args):
    
    
    args.lr = args.start_lr * 0.5 * (1.0 + math.cos(math.pi * args.steps / args.total_steps ))
    
    for param_group in optimizer.param_groups:
        
        if param_group["freeze_lr"] == True:
            param_group["lr"] = args.start_lr #freeze predictor lr

        else:
            param_group["lr"] = args.lr
