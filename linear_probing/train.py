import torch
import torch.distributed as dist
import utils.optim as optim
import time


def train_one_epoch(encoder, optimizer, train_loader, criterion, logger, args):
    
    encoder.eval() #BN freeze
    optimizer.zero_grad()
    
    start_dt_time = time.time()

    for idx, (inputs, targets) in enumerate(train_loader):
        
        end_dt_time = time.time()
        

        bs = inputs.size(0)

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optim.adjust_lr(optimizer, args) #cosin, no warmup

        optimizer.zero_grad()

        
        start_forward_time = time.time()
        out = encoder(inputs) #[bs, n_classes]
        end_forward_time = time.time()

        loss = criterion(out, targets)/bs


        loss.backward()
        optimizer.step()

        acc = (out.max(1)[1] == targets).sum()/bs

        
        
        args.steps += 1
        
        
        if logger != None:
            
            logger.add_scalar('/train/loss', loss.item(), args.steps*args.batch_size) #loss vs nb sample
            logger.add_scalar('/train/acc', acc.item(), args.steps*args.batch_size) #loss vs nb sample
            logger.add_scalar('/train/lr', args.lr, args.steps*args.batch_size)
            logger.add_scalar('/train/data_time', end_dt_time - start_dt_time, args.steps*args.batch_size)
            logger.add_scalar('/train/forward_time', end_forward_time - start_forward_time, args.steps*args.batch_size)


        start_dt_time = time.time()



def validation(encoder, test_loader, criterion, logger, args):


    encoder.eval()
    
    torch.cuda.synchronize()
            
    validation_loss = 0.0
    total_valid = torch.zeros(1).cuda()
    total_examples = torch.zeros(1).cuda()
    total_loss = torch.zeros(1).cuda()

    
            
    for idx, (inputs, targets) in enumerate(test_loader): #[bs, C, H, W]

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled = True): #fp16

                out = encoder(inputs)
            
        loss = criterion(out, targets)

        total_loss += loss
        total_examples += inputs.size(0)
        total_valid += (out.max(1)[1] == targets).sum()

    #synchro loss

    torch.cuda.synchronize()
            
    dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_valid, op=dist.ReduceOp.SUM)

    if logger != None:
        logger.add_scalar('/validation/acc', total_valid.item()/total_examples.item(), args.steps*args.batch_size)
        logger.add_scalar('/validation/loss', total_loss.item()/total_examples.item(), args.steps*args.batch_size)
    
    return total_valid.item()/total_examples.item()