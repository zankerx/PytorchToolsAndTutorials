import numpy as np

import torch
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import parser
import models
import train

import utils.slurmEnvironement as slurm
import utils.tools as tools
import utils.transforms as transforms
import utils.optim as optim

import time

def main(args):
    

    # setup ddp :
    args.world_size = slurm.getWorldSize()
    args.rank = slurm.getRank()
    args.local_rank = slurm.getLocalRank()
    args.num_workers = slurm.getNumWorker()
    slurm.setupEnvironement()#Environelent var : adresse/port du noeud maitre/world_size/rank
    
    # initialisation de la communication via variables d'environement
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # random, reproductibilité
    torch.manual_seed(args.seed) #fixe aussi la seed cuda
    np.random.seed(args.seed)
    
    # set gpu
    torch.cuda.set_device(args.local_rank) #[0-n] par noeud
    
    batch_size_per_gpu = args.batch_size // args.world_size
    
    args.job_id = slurm.getJobId() #nom des logs

    #log on master
    if args.rank == 0:
        #tensorboard Logs
        logger = SummaryWriter(log_dir='./runs/{}'.format(args.job_id))
        
    else:
        logger = None
    

    # setup transforms
    transforms_pretrain = transforms.TransformsSimSiam(args)
    transforms_cifar = transforms.TransformsCifar(args)
    
    
    # data pre-train
    if args.data_path == 'CIFAR10':
        dataset_train = datasets.CIFAR10('./data', train=True, transform=transforms_pretrain)
    else:
        dataset_train = datasets.ImageFolder(args.data_path, transform = transforms_pretrain)

    # setup data CIFAR --> knn
    if args.knn_train_data_path == 'CIFAR10':
        dataset_train_CIFAR10 = datasets.CIFAR10('./data', train=True, transform=transforms_cifar)
        dataset_test_CIFAR10 = datasets.CIFAR10('./data', train=False, transform=transforms_cifar)
    else:
        dataset_train_CIFAR10 = datasets.ImageFolder(args.knn_train_data_path, transform=transforms_cifar)
        dataset_test_CIFAR10 = datasets.ImageFolder(args.knn_test_data_path, transform=transforms_cifar)
        
    #data ddp
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    
    sampler_train_CIFAR10 = torch.utils.data.DistributedSampler(dataset_train_CIFAR10, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    sampler_test_CIFAR10 = torch.utils.data.DistributedSampler(dataset_test_CIFAR10, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    
    
    #setup dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    
    
    train_loader_CIFAR10 = torch.utils.data.DataLoader(
        dataset_train_CIFAR10, sampler=sampler_train_CIFAR10,
        batch_size=int(batch_size_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    
    test_loader_CIFAR10 = torch.utils.data.DataLoader(
        dataset_test_CIFAR10, sampler=sampler_test_CIFAR10,
        batch_size=int(batch_size_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    
    n_step_epoch = len(train_loader)
    args.total_steps = n_step_epoch*args.epochs


    #setup model

    encoder_without_ddp = models.EncoderCNN(args)
    predictor_without_ddp = models.Predictor(args)

    #BN synchro GPU
    encoder_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder_without_ddp)
    predictor_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor_without_ddp)

    #cpu -> gpu
    encoder_without_ddp.cuda()
    predictor_without_ddp.cuda()

    #setup model DDP 
    encoder = torch.nn.parallel.DistributedDataParallel(encoder_without_ddp, device_ids=[args.local_rank], find_unused_parameters=True)
    predictor = torch.nn.parallel.DistributedDataParallel(predictor_without_ddp, device_ids=[args.local_rank], find_unused_parameters=True)
    
    #lr constant sur le predicteur, variable sur l'encoder, gestion du lr dans optim.py
    params_groups = param_groups = [
                                    {
                                        'params': [p for n, p in encoder_without_ddp.named_parameters()],
                                        'freeze_lr' : False
                                    },
                                    {
                                        'params': [p for n, p in predictor_without_ddp.named_parameters()],
                                        'freeze_lr' : True
                                    }
                                   ]
    
    args.start_lr = args.start_lr * args.batch_size/256 #b_lr * bs/256
    
    optimizer = torch.optim.SGD(params_groups, lr=args.start_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
    criterion = torch.nn.CosineSimilarity().cuda()

    #training loop:
    
    accs = [] #log best acc

    for e in range(1, args.epochs + 1):
        
        start_time = time.time()
        
        train_loader.sampler.set_epoch(e)#random suffle
        
        #train
        train.train_one_epoch(encoder, predictor, optimizer, criterion, train_loader, logger, args)
        
        #validation
        acc = train.validation(encoder, train_loader_CIFAR10, test_loader_CIFAR10, logger, args)
        accs.append(acc)

        elapsed_time = time.time() - start_time

        #log console/ckpt
        if (e % args.ckpt_interval == 0) and (args.rank == 0):
            tools.save_ckpt(encoder_without_ddp, predictor_without_ddp, optimizer, e, args)
        
        if logger != None:
            print('Epoch : {} | Acc : {:.2f} | BestAcc : {:.2f} | time : {:.2f}'.format(e, acc, max(accs), elapsed_time))

    print('end training')

    if args.rank == 0:
        
        tools.save_ckpt(encoder_without_ddp, predictor_without_ddp, optimizer, e, args)
        tools.save_hparams(logger, args, max(accs))
        
    
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
