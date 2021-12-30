import argparse
import torch.nn.parallel
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pickle
import os
import csv
import json

import dataset as D
from models import densenet, wideresnet, resnet
import utils


parser = argparse.ArgumentParser()

parser.add_argument('--val_in_dataset', default='./data/tomato_data/val_data/in', type=str, help='path to validation in dataset : ./data/tst_data/CIFAR10 | ./data/tst_data/CIFAR100')
parser.add_argument('--val_out_dataset', default='./data/tomato_data/val_data/out', type=str, help='path to validation ood dataset : ./data/val_data/iSUN')
parser.add_argument('--train_result_path', default='./train_result', type=str, help='path to train result')
parser.add_argument('--model', default='resnet50', type=str, help='model architecture : wide | dense | resnet50')
parser.add_argument('--save_path', default='./validation_result', type=str, help='Path to save result')
parser.add_argument('--max_fold', type=int, default = 5, help='Number of classifiers to ensemble')

parser.add_argument('--batchSize', type=int, default=32, help='batch size')


args = parser.parse_args()
print(args)

def main():

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.save_path+'/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load validation data ( in + ood )
    val_id_dataset = D.get_dataset(args.val_in_dataset, fold=0)
    val_ood_dataset = D.get_dataset(args.val_out_dataset, fold=0)

    val_dataset = D.ValDataset(val_id_dataset, val_ood_dataset, in_class=None)

    val_loader = DataLoader(val_dataset,batch_size=args.batchSize, shuffle=False, num_workers=4)

    num_class = int(max(val_dataset.in_dataset.targets)+1)

    criterion = nn.CrossEntropyLoss().cuda()

    temperatures = [1, 10, 100, 1000]
    epsilons = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    # CSV file to record performances
    f = open(args.save_path + '/validation_result_model{}.csv'.format(args.model), 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['temperature','epsilon','FPR','Detection err','AUROC','AUPR_In','AUPR_Out'])

    # Validation
    for temperature in temperatures:
        for epsilon in epsilons:
            for fold in range(1,args.max_fold+1):
                # Decide model
                if args.model == 'wide':
                    net = wideresnet.WideResNet(int(num_class * 4 / 5))
                elif args.model == 'dense':
                    net = densenet.DenseNet(int(num_class * 4 / 5))
                elif args.model == 'resnet50':
                    net = resnet.ResNet50(num_classes = num_class-1)

                net = nn.DataParallel(net).cuda()
                state_dict = torch.load(args.train_result_path + '/{model}_fold_{fold}/model_state_dict.pth'.format(model=args.model, fold=fold))
                net.load_state_dict(state_dict)
                net.eval()
                in_class = torch.load(args.train_result_path + '/{model}_fold_{fold}/in_class.pt'.format(model=args.model, fold=fold))

                # Validate
                validate(net, val_loader, num_class,temperature, epsilon,in_class,criterion,fold)

            # Get ood detection performance
            performance = utils.ood_detection(len(val_id_dataset),len(val_ood_dataset),temperature,epsilon,num_class,args.save_path,args.max_fold)
            # Record performance on (temperature, epsilon)
            wr.writerow([temperature,epsilon]+performance)
            print("Temperature : {0} Epsilon : {1} FPR : {2:.4f} Detection err : {3:.4f} AUROC : {4:.4f} AUPRIN : {5:.4f} AUPROut : {6:.4f} "
                  .format(temperature,epsilon,performance[0],performance[1],performance[2],performance[3],performance[4]))

    f.close()


def validate(net,val_loader,num_class,temperature,epsilon,in_class, criterion,fold):
    t0 = time.time()

    #nsplit = int(num_class * 0.8)
    nsplit = num_class-1

    in_sfx = np.array([])
    in_pro = np.array([])
    out_sfx = np.array([])
    out_pro = np.array([])
    ######################################In distribution data########################################
    for j, data in enumerate(val_loader):
        images, targets = data

        inputs = images.cuda().requires_grad_()
        outputs = net(inputs)

        o_output = np.zeros((images.size()[0], num_class))

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        # nnOutputs = outputs.detach().cpu()
        # nnOutputs = nnOutputs.numpy()
        # nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        # stack softmax output batch_size x num_class (for out_class, give 0)

        for i in range(images.shape[0]):
            if targets[i]>=0:
                in_sfx = np.vstack((in_sfx, o_output[i])) if in_sfx.size else o_output[i]
            else:
                out_sfx = np.vstack((out_sfx, o_output[i])) if out_sfx.size else o_output[i]

        # if j<int(len(val_loader)*0.5):
        #     in_sfx = np.vstack((in_sfx, o_output)) if in_sfx.size else o_output
        # else:
        #     out_sfx = np.vstack((out_sfx, o_output)) if out_sfx.size else o_output

        o_output = np.zeros((images.size()[0], num_class))

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)

        # Make gradient 1 or -1
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[:, 0] = (gradient[:, 0]) / 0.5
        #gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
        #                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / 0.5)
        # gradient[:, 1] = (gradient[:, 1]) / (stdv[1])
        # gradient[:, 2] = (gradient[:, 2]) / (stdv[2])
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)

        outputs = net(tempInputs)
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations

        #nnOutputs = outputs.detach().cpu()
        #nnOutputs = nnOutputs.numpy()
        #nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        nnOutputs = nn.functional.softmax(outputs,dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        # Stack Temperature scaled softmax probability on perturbed input

        for i in range(images.shape[0]):
            if targets[i]>=0:
                in_pro = np.vstack((in_pro, o_output[i])) if in_pro.size else o_output[i]
            else:
                out_pro = np.vstack((out_pro, o_output[i])) if out_pro.size else o_output[i]

        # if j<int(len(val_loader)*0.5):
        #     in_pro = np.vstack((in_pro, o_output)) if in_pro.size else o_output
        # else:
        #     out_pro = np.vstack((out_pro, o_output)) if out_pro.size else o_output

        print("Validate on Temperature : {} Epsilon : {}   [{:4}/{:4}] images processed, {:.1f} seconds used.".format(temperature,epsilon,j + 1, len(val_loader),time.time() - t0))
        t0 = time.time()

    print("Classifier {0} Done".format(fold))
    data = {'in_sfx': in_sfx, 'in_pro': in_pro, 'out_sfx': out_sfx, 'out_pro': out_pro}
    pickle.dump(data, open(args.save_path + "/{temp}_{epsilon}_{fold}.p".format(temp=temperature,epsilon=epsilon,fold=fold), "wb"))

if __name__ == "__main__":
    main()
