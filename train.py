from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time
import dataloader
import dataset as D
from models import densenet, wideresnet,resnet
import numpy as np
from itertools import cycle
from utils import Logger, AverageMeter, draw_curve
import metrics
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet50', type=str, help='model architecture : wide | dense | resnet50')
parser.add_argument('--train_dataset', default='./data/tomato_data/trn_data', type=str, help='path to training dataset : ./data/trn_data/CIFAR10 | ./data/trn_data/CIFAR100')
parser.add_argument('--val_in_dataset', default='./data/tomato_data/val_data/in', type=str, help='path to validation dataset : ./data/val_data/iSUN')
parser.add_argument('--val_out_dataset', default='./data/tomato_data/val_data/out', type=str, help='path to validation dataset : ./data/val_data/iSUN')
parser.add_argument('--save_path', default='./tomato_result_exp1_beta1', type=str, help='Path to save result')
parser.add_argument('--batchSize', type=int, default=32, help='batch size')

parser.add_argument('--max_fold', type=int, default = 6, help='Number of classifiers to ensemble')
parser.add_argument('--epoch', type=int, default=150, help='number of epochs to train for (default : 100)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.1')
parser.add_argument('--final_lr', type=float, default=0.0001, help='final learning rate, default=0.0001')
parser.add_argument('--momentum', default=0.9, type=float,help='momentum for SGD optimizer (default:0.9)')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--beta', type=float, default=1,help='coefficient of margin entropy loss (default: 0.2)')
parser.add_argument('--margin', default=0.4, type=float, help='Least margin between avg entropy of id&ood (default : 0.4)')

args = parser.parse_args()
print(args)


def main():
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    avg_performance = 0.0

    for fold in range(1,args.max_fold+1):
        print("Train start on fold {0}".format(fold))

        if not os.path.exists(save_path+'/{0}_fold_{1}'.format(args.model,fold)):
            os.makedirs(save_path+'/{0}_fold_{1}'.format(args.model,fold))

        work_dir = save_path+'/{0}_fold_{1}'.format(args.model,fold)

        # Train Dataset ( in / out )
        train_id_dataset, train_ood_dataset, in_class, num_class = D.get_dataset(args.train_dataset, fold)
        # Validation Dataset ( in / out )
        val_id_dataset = D.get_dataset(args.val_in_dataset,fold=0)
        val_ood_dataset = D.get_dataset(args.val_out_dataset, fold=0)

        val_dataset = D.ValDataset(val_id_dataset,val_ood_dataset, in_class)

        id_train_loader = DataLoader(train_id_dataset,batch_size=args.batchSize, shuffle=True, num_workers=4)
        ood_train_loader = DataLoader(train_ood_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset,batch_size=args.batchSize,shuffle=False)

        # Save in classes
        torch.save(in_class, work_dir+'/in_class.pt'.format(args.model,fold))

        # Set model
        if args.model == 'wide':
            model = wideresnet.WideResNet(num_classes = int(num_class*4/5), dropRate=0.3)
        elif args.model == 'dense':
            model = densenet.DenseNet(num_classes = int(num_class*4/5))
        elif args.model == 'resnet50':
            model = resnet.ResNet50(num_classes = num_class-1)

        model = nn.DataParallel(model).cuda()

        # Set optim & loss
        #gamma = ((1/args.lr)*args.final_lr)**(1/args.epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr= args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov=True)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80,120],gamma=0.1)
        criterion = nn.CrossEntropyLoss().cuda()

        # Set logger
        train_logger = Logger(os.path.join(work_dir, 'train.log'))
        val_logger = Logger(os.path.join(work_dir, 'val.log'))

        best_auroc1 = 0.0

        # Train & Validate on fold k
        for epoch in range(args.epoch):
            train(model,id_train_loader,ood_train_loader,optimizer,criterion,epoch,train_logger)
            best_auroc1 = validate(model,val_loader,criterion,epoch,work_dir,val_logger,best_auroc1)
            scheduler.step()

        print("\nTrain on fold {0} Done".format(fold))

        # Draw curves
        draw_curve(work_dir,train_logger,val_logger)

        avg_performance += best_auroc1

    avg_performance = avg_performance / args.max_fold
    os.rename(save_path, save_path + '_{perf:.4f}'.format(perf=avg_performance))

def train(model, id_train_loader, ood_train_loader, optimizer, criterion, epoch, train_logger):
    model.train()

    iteration_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = AverageMeter()

    end = time.time()
    for i, ((data_id,target_id),(data_ood,_)) in enumerate(zip(id_train_loader,cycle(ood_train_loader))):
        data_time.update(time.time()-end)

        data = torch.cat((data_id,data_ood),axis=0)

        data, target_id = data.cuda(), target_id.cuda()

        # Calculate output
        output = model(data)
        output_id = output[:data_id.shape[0]]
        output_ood = output[data_id.shape[0]:]

        # Average entropy of id & ood
        Entropy_id = -torch.mean(torch.sum(F.log_softmax(output_id, dim=1) * F.softmax(output_id, dim=1), dim=1))
        Entropy_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

        # Loss
        loss = criterion(output_id, target_id) + args.beta * torch.clamp(args.margin + Entropy_id - Entropy_ood, min=0)
        train_loss.update(loss.item(), data_id.shape[0])

        # Update Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration_time.update(time.time()-end)
        end = time.time()
        if i%10 ==0 and i != 0:
            print('[{0}/{1}][{2}/{3}] Train Loss : {loss.val:.4f} ({loss.avg:.4f}) Iteration Time : {iteration_time.val:.3f} ({iteration_time.avg:.3f}) Data Time : {data_time.val:.3f} ({data_time.avg:.3f})'
                  .format(epoch+1, args.epoch,i,len(id_train_loader),loss=train_loss,iteration_time=iteration_time, data_time = data_time))

    train_logger.write([epoch, train_loss.avg])


def validate(model, val_loader, criterion, epoch, work_dir,val_logger,best_auroc1):
    model.eval()
    gts, probs = [],[]

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = F.softmax(model(data),dim=1)
            prob = output

            for i in range(len(prob)):
                gts.append(target[i].item())
                probs.append(prob[i].cpu().numpy())

    # Get auroc score
    cifar = []
    other = []
    for i in range(len(gts)):
        gt = gts[i]
        prob = probs[i]
        if gt >= 0:
            cifar.append(np.max(prob))
        else:
            other.append(np.max(prob))
    cifar, other = np.array(cifar), np.array(other)

    auroc = metrics.get_auroc(cifar,other)

    print("\nEpoch : {0}    Validation AUROC: {1:.4f} \n".format(epoch+1,auroc))

    is_best = auroc > best_auroc1
    if is_best:
        best_auroc1 = auroc
        torch.save(model.state_dict(), work_dir+'/model_state_dict.pth')

    val_logger.write([epoch, auroc])

    return best_auroc1

if __name__ == "__main__":
    main()
