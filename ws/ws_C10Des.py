from __future__ import print_function
import os ,shutil ,time ,random ,  math ,argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import torch.nn.functional as F
from utils import Bar,   trainTdp_bound_predic, testTdp , saveVitData 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    
#  tdp Logit Params  
parser.add_argument('-r', '--randStd', default=0.0, type=float)
parser.add_argument('-s', '--senBound', default=0.001, type=float)
parser.add_argument('-z', '--ABgroup', default=0, type=float)

# Datasets
parser.add_argument('-g', '--gpuDevice', default='-1', type=str)
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=80000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
# main parameters 0.12 sgd 0.022 adma
parser.add_argument('--lr', '--learning-rate', default=0.022, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.991, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

# n = (depth - 4) / 3
parser.add_argument('--depth', type=int, default=22, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
randStd  = float(args.randStd) 

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuDevice
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda: torch.cuda.manual_seed_all(args.manualSeed)

EMBSIZE = 69 
PRESIZE = 10 
save_result_dir = '/DIR/c10/wsDes/'

def main():
    global best_acc
    # Data  
    LoadData = loadCifar10(Test=True , ABgroup = float(args.ABgroup ))
    LoadData.weightMatrix = [] ; LoadData.smuWM =[]
    num_classes = 10 
    # Model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
        
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # criterion = nn.CrossEntropyLoss()
    criterion = None 
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('*' * 100)
        print("Train New Model ! ")
        print('*' * 100)

    print(' Train Logit Noise * weight -- 0 ')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        train_loss, train_acc = trainTdp_bound_predic(LoadData, model, 
            criterion, optimizer, epoch ,args.train_batch ,randStd , numclass=10 ,senBound=args.senBound  )

        if epoch % 20 == 0 :
            test_loss, test_acc   =  testTdp(LoadData, model, criterion, epoch, use_cuda ,batch_size =args.train_batch  )

            print(epoch , ' TrainAcc.', train_acc , ' testAcc,', test_acc , ' loss1,',train_loss , ' loss2,', test_loss )
                
        if epoch >  180 : 
            saveVitData(LoadData, model   , args.train_batch, senBound= args.senBound  \
                    , sp= save_result_dir , randStd=randStd  , embSize = EMBSIZE ,  predSize= PRESIZE , ABgroup = args.ABgroup )
            
def adjust_learning_rate(optimizer, epoch):
    global state
    # if epoch in args.schedule:
    if True:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
if __name__ == '__main__':
    main()
