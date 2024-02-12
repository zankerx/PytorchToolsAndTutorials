import argparse




def parse_args():
    
    parser = argparse.ArgumentParser("linear Probing")

    #ckpt pretrain model
    parser.add_argument("--architecture", default='resnet50', type=str, help="resnet50, resnet18, etc.")
    parser.add_argument("--ckpt_path", default='./ckpt_100e', type=str, help="cktp path")
    parser.add_argument("--data_train_path", default='/gpfsdswork/dataset/imagenet/train', type=str, help="img path")
    parser.add_argument("--data_test_path", default='/gpfsdswork/dataset/imagenet/val', type=str, help="img path")

    parser.add_argument("--n_classes", default=1000, type=int, help="nombre de classes")

    #global training parameters
    
    parser.add_argument("--ckpt_interval", default=100, type=int, help="checkpoint interval")
    parser.add_argument("--epochs", default=90, type=int, help="Nombre d'epochs total")
    parser.add_argument("--batch_size", default=4096, type=int, help="batch_size globale")
    
    #optim params
    parser.add_argument("--start_lr", default=0.1, type=float, help="lr initial (base_lr*bs/256)")#paper 0.02
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    
    
    
    #training environement parameters
    parser.add_argument("--seed", default=123, type=int, help="random seed")
    parser.add_argument("--pin_mem", default=True, type=bool, help="using pin memory")
    parser.add_argument("--world_size", default=1, type=int, help="world_size")
    parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
    parser.add_argument("--rank", default=0, type=int, help="global rank")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank")

    #Other
    parser.add_argument("--total_steps", default=0, type=int, help="total_steps")
    parser.add_argument("--steps", default=0, type=int, help="steps")
    parser.add_argument("--lr", default=0, type=float, help="lr")
    parser.add_argument("--wd", default=0, type=float, help="wd")
    parser.add_argument("--job_id", default=0, type=int, help="job_id")

    return parser.parse_args()