import argparse


def parse_args():
    
    #default : cifar10 experiment

    parser = argparse.ArgumentParser("training simsiam")

    #model params
    parser.add_argument("--architecture", default='resnet18', type=str, help="resnet50, resnet18, etc.")
    parser.add_argument("--emb_dim", default=2048, type=int, help="dim de la représentation")
    parser.add_argument("--pred_dim", default=512, type=int, help="predictor dim bottleneck")

    #global training parameters
    parser.add_argument("--ckpt_interval", default=100, type=int, help="checkpoint interval")
    parser.add_argument("--epochs", default=800, type=int, help="Nombre d'epochs total")
    parser.add_argument("--batch_size", default=512, type=int, help="batch_size globale")
    parser.add_argument("--n_knn", default=200, type=int, help="n knn")
    parser.add_argument("--n_classes_knn", default=10, type=int, help="n classes knn")
    
    #optim params
    parser.add_argument("--start_lr", default=0.03, type=float, help="lr initial (base_lr*bs/256)")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="SGD weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    
    #data params
    parser.add_argument("--img_size", default=32, type=int, help="img size")
    parser.add_argument("--data_path", default='CIFAR10', type=str, help="img path")
    parser.add_argument("--knn_train_data_path", default='CIFAR10', type=str, help="img path")
    parser.add_argument("--knn_test_data_path", default='CIFAR10', type=str, help="img path")
    
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