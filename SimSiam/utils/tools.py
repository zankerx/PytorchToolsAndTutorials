import torch

#checkpoints

def save_ckpt(encoder, predictor, optimizer, e, args):
    
    state = {'encoder' : encoder.state_dict(),
             'predictor' : predictor.state_dict(),
             'optimizer' : optimizer.state_dict(),
             'epoch' : e,
             'args' : vars(args)
            }
        
    torch.save(state, 'ckpt_{}_{}'.format(e, args.job_id))


def load_ckpt(path, encoder, predictor, optimizer):
    
    state = torch.load(path)
    
    encoder.load_state_dict(state['encoder'])
    predictor.load_state_dict(state['predictor'])
    optimizer.load_state_dict(state['optimizer'])
    
    return state['epoch']


#hparams + metriques
def save_hparams(logger, args, best_acc):
    
    hparams_dict = vars(args)
    metric_dict = {'BestAccuracy' : best_acc}

    logger.add_hparams(hparams_dict, metric_dict)
    