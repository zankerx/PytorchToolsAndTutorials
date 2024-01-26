import torch
import torch.nn as nn



class DropPath(nn.Module):
    
    def __init__(self, p: float = 0.0, rescale: bool = True):
        super(DropPath, self).__init__()
        
        self.p = p
        self.rescale = rescale
    
    def drop_path(self, x: torch.Tensor, p: float = 0.0, rescale: bool = True, training: bool = False):
    
        #check training or proba == 0.0
        if p == 0.0 or not training:
            return x
        
        keep_proba = 1.0 - p
        
        N, L, D = x.size()
        
        #drop mask
        drop = torch.empty((N,L), device=x.device).bernoulli_(keep_proba).unsqueeze(-1)
        
        if rescale == True:
            
            drop.div_(keep_proba)
        
        return x*drop

    def forward(self, x):
        
        return self.drop_path(x, self.p, self.rescale, self.training)


class Mlp(nn.Module):
    
    def __init__(self, 
                 inputs_dim = 768,
                 hidden_dim = 3072,
                 outputs_dim = 768,
                 activation = nn.GELU,
                 drop = 0.1,
                 bias = True):
        
        
        
        super(Mlp, self).__init__()
        
        self.fc1 = nn.Linear(inputs_dim, hidden_dim, bias=bias)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim, bias=bias)
        self.drop = nn.Dropout(drop)
        
    
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.activation(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        
        return out
