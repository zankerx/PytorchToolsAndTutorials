from json import tool
import parser
import os
import time

import torchvision.datasets as datasets

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import torchvision.models as models
import torchvision.transforms as transforms

import train

import utils.slurmEnvironement as slurm
import utils.loss_scaler as loss_scaler

import utils.optim as optim
import utils.mask_collate as mask_collate

from apex.parallel.LARC import LARC



def main(args):
    
    # setup ddp :
    args.world_size = slurm.getWorldSize()
    args.rank = slurm.getRank()
    args.local_rank = slurm.getLocalRank()
    args.num_workers = slurm.getNumWorker()
    slurm.setMaster()#adresse du noeud maitre
    
    # initialisation de la communication
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    
    # random, reproductibilité
    torch.manual_seed(args.seed) #fixe aussi la seed cuda
    np.random.seed(args.seed)
    
    # set gpu
    torch.cuda.set_device(args.local_rank)
    
    batch_size_per_gpu = args.batch_size // args.world_size
    
    args.job_id = slurm.getJobId() #nom des logs

    #log on master
    if args.rank == 0:
        logger = SummaryWriter(log_dir='./runs/{}'.format(args.job_id))
        
    else:
        logger = None
    
    
    # setup transforms
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    # setup data
    dataset_train = datasets.ImageFolder(args.data_train_path, transform=transforms_train)
    dataset_test = datasets.ImageFolder(args.data_test_path, transform=transforms_test)
    
    #data ddp
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    
    
    #setup dataloaders

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=int(batch_size_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=int(batch_size_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    
    n_step_epoch = len(train_loader)
    args.total_steps = n_step_epoch*args.epochs

    #setup model

    model = models.__dict__[args.architecture](num_classes = args.n_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if logger != None:
        print(model)

    checkpoint = torch.load(args.ckpt_path)


    #select params from ckpt : https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py

    state_dict = checkpoint['encoder']
    for k in list(state_dict.keys()):
        if k.startswith('encoder') and not k.startswith('encoder.fc'):
            state_dict[k[len("encoder."):]] = state_dict[k]
            
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)

    if logger != None:
        print(msg)

    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}



    #cpu -> gpu
    model.cuda()

    #setup model DDP 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True)
    


    
    args.start_lr = args.start_lr * args.batch_size/256 #b_lr * bs/256
    

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias


    optimizer = torch.optim.SGD(parameters, lr=args.start_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum').cuda()

    #training loop:
    
    accs = [] #log best acc

    

    for e in range(1, args.epochs + 1):
        
        start_time = time.time()
        
        train_loader.sampler.set_epoch(e)
        
        #train
        train.train_one_epoch(model, optimizer, train_loader, criterion, logger, args)
        
        #validation
        acc = 100*train.validation(model, test_loader, criterion, logger, args)

        end_time = time.time()

        accs.append(acc)

        #log console/ckpt
        if logger != None:
            print('Epoch : {} | Acc : {:.2f}% | Best Acc : {:.2f}% | time : {:.2f}'.format(e, acc, max(accs), end_time - start_time))
        
            

    if logger != None:
        print('end training')
        
    '''
    if args.rank == 0:
        
        tools.save_ckpt(encoder_without_ddp, predictor_without_ddp, optimizer, scaler, e, args)
        tools.save_hparams(logger, args, max(accs))
        
    '''
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
