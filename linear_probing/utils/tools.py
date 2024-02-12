import torch

#checkpoints

def save_ckpt(model, predictor1, optimizer, scaler, e, args):
    
    state = {'model' : model.state_dict(),
             'predictor1' : predictor1.state_dict(),
             'optimizer' : optimizer.state_dict(),
             'scaler' : scaler.state_dict(),
             'epoch' : e,
             'args' : vars(args)
            }
        
    torch.save(state, 'ckpt_{}_{}'.format(e, args.job_id))


def load_ckpt(path, model, model_ema, predictor, optimizer, scaler):
    
    state = torch.load(path)
    
    model.load_state_dict(state['model'])
    model_ema.load_state_dict(state['model_ema'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    
    return state['epoch']


#hparams + metriques
def save_hparams(logger, args, best_acc):
    
    hparams_dict = vars(args)
    metric_dict = {'BestAccuracy' : best_acc}

    logger.add_hparams(hparams_dict, metric_dict)
    