from turtle import forward
import torch.nn as nn
import torch
import numpy as np
import torchvision
import resnet

'''

Encoder d'images et prédicteur temporel (causal type LLM)

'''

class EncoderCNN(nn.Module):

    def __init__(self, args):

        super(EncoderCNN, self).__init__()

        #self.encoder = torchvision.models.__dict__[args.architecture](num_classes=args.emb_dim, zero_init_residual = True)

        self.encoder = resnet.resnet18()
        prev_dim = self.encoder.fc.weight.shape[1]
        


        #2 layers dans le cas cifar10
        '''
        
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
        '''
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, args.emb_dim, bias=False),
                                        nn.BatchNorm1d(args.emb_dim, affine=False)) # output layer

    def forward(self, x):
        
        # x : frames de la video

        x = self.encoder(x)

        return x


class Predictor(nn.Module):
    
    #VIT version AVG pool
    
    def __init__(self, args):
        super(Predictor, self).__init__()

        self.net = nn.Sequential(nn.Linear(args.emb_dim, args.pred_dim, bias=False),
                                nn.BatchNorm1d(args.pred_dim),
                                nn.ReLU(),
                                nn.Linear(args.pred_dim, args.emb_dim))

    def forward(self, x):
        return self.net(x)




