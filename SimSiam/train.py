import torch
import torch.distributed as dist
import utils.optim as optim



def knn_predict(feature, feature_bank, feature_labels, classes = 10, knn_k = 200, knn_t = 0.1):
    
    # from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    
    #feature : [n_test, dim]
    #feature_bank : [n_train, dim]
    #feature_labels : [n_train]
    #classes : nb classes
    #knn_k : nb voisins
    #knn_t : temperature

    sim_matrix = torch.mm(feature, feature_bank.T) #[n_test, n_train]

    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) #[n_test, knn_k], [n_test, knn_k]

    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)

    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels

def train_one_epoch(encoder, predictor, optimizer, criterion, train_loader, logger, args):
    
    encoder.train()
    predictor.train()
    optimizer.zero_grad()
    
    for idx, (x, _) in enumerate(train_loader):
        
        x1 = x[0]
        x2 = x[1]

        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        optim.adjust_lr(optimizer, args) #cosin, no warmup

        optimizer.zero_grad()

            
        emb1 = encoder(x1) #[bs, dim]
        emb2 = encoder(x2)

        pred1 = predictor(emb1)
        pred2 = predictor(emb2)

        loss1 = criterion(pred1, emb2.detach()).mean()
        loss2 = criterion(pred2, emb1.detach()).mean()

        loss = -0.5*(loss1 + loss2)

        loss.backward()
        optimizer.step()

        args.steps += 1
        
        #log on master
        if logger != None:
            
            logger.add_scalar('/train/loss', loss.item(), args.steps*args.batch_size) #loss vs nb sample
            logger.add_scalar('/train/lr', args.lr, args.steps*args.batch_size)




def validation(encoder, train_loader, test_loader, logger, args):
    
    #on peut améliorer le calcul du knn mais osef
    encoder.eval()
        
    

    #compute on emb with all gpus
    #knn emb

    emb_train = []
    lab_train = []
    emb_test = []
    lab_test = []

    with torch.no_grad():
        
        #train
        for idx, (inputs, label) in enumerate(train_loader): #[bs, C, H, W]

            inputs = inputs.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)


            emb_train.append(encoder(inputs))
            lab_train.append(label)
        
        #test
        for idx, (inputs, label) in enumerate(test_loader): #[bs, C, H, W]

            inputs = inputs.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
                
            emb_test.append(encoder(inputs))
            lab_test.append(label)

    emb_train = torch.cat(emb_train, dim = 0)
    lab_train = torch.cat(lab_train, dim = 0)
    emb_test = torch.cat(emb_test, dim = 0)
    lab_test = torch.cat(lab_test, dim = 0)

    #normalisation sur la sphère

    emb_train = torch.nn.functional.normalize(emb_train)
    emb_test = torch.nn.functional.normalize(emb_test)
    
    #print('emb_train : {} | {}'.format(emb_train.size(), args.rank))

    #all gather
    all_emb_train = [torch.zeros_like(emb_train) for _ in range(args.world_size)]
    all_emb_test = [torch.zeros_like(emb_test) for _ in range(args.world_size)]
    all_lab_train = [torch.zeros_like(lab_train) for _ in range(args.world_size)]
    all_lab_test = [torch.zeros_like(lab_test) for _ in range(args.world_size)]

    dist.all_gather(all_emb_train, emb_train)
    dist.all_gather(all_emb_test, emb_test)
    dist.all_gather(all_lab_train, lab_train)
    dist.all_gather(all_lab_test, lab_test)

    all_emb_train = torch.cat(all_emb_train)
    all_emb_test = torch.cat(all_emb_test)
    all_lab_train = torch.cat(all_lab_train)
    all_lab_test = torch.cat(all_lab_test)

    #print('emb_train : {} | {}'.format(all_emb_train.size(), args.rank))

    #knn on master
    acc = 0.0

    if logger != None:  
        pred = knn_predict(all_emb_test, all_emb_train, all_lab_train, args.n_classes_knn, args.n_knn)

        acc = (pred[:,0] == all_lab_test).sum().item()/len(all_lab_test)

        logger.add_scalar('/validation/AccuracyKNNCifar10', acc, args.steps*args.batch_size)
    
    return acc