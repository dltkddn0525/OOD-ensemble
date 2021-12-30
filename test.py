import os
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import pickle
from models import densenet, wideresnet, resnet
import dataset as D
import metrics
import json
import sklearn

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--tst_in_dataset', default='./data/tomato_exp2_data/ood_tst_data/in', type=str, help='path to in dataset : ./data/tst_data/CIFAR10 | ./data/tst_data/CIFAR100')
parser.add_argument('--tst_out_dataset', default='./data/tomato_exp2_data/ood_tst_data/out', type=str, help='path to ood dataset : ./data/tst_data/Imagenet')
parser.add_argument('--cls_test_dataset', default='./data/tomato_exp2_data/cls_tst_data', type=str, help='path to classification test dataset : ./data/tst_data/Imagenet')
parser.add_argument('--epsilon', default=0.002, type=float,help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int, help='temperature scaling')
parser.add_argument('--model', default='resnet50', type=str, help='wide | dense | resnet50')
parser.add_argument('--save_path', default='./test_result', type=str, help='save path')
parser.add_argument('--train_result_path', default='./train_result', type=str, help='path to saved models')
parser.add_argument('--max_fold', type=int, default =5, help='Number of classifiers to ensemble')
parser.add_argument('--batchSize', type=int, default=32, help='batch size')

args = parser.parse_args()
print(args)
def main():
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    criterion = nn.CrossEntropyLoss().cuda()

    #Load Dataset & Dataloader
    testsetIn = D.get_dataset(args.tst_in_dataset, fold=0)
    testsetOut = D.get_dataset(args.tst_out_dataset, fold=0)
    cls_testset = D.get_dataset(args.cls_test_dataset, fold=0)

    testloaderIn = torch.utils.data.DataLoader(testsetIn, batch_size=args.batchSize, shuffle=False, num_workers=2)
    testloaderOut = torch.utils.data.DataLoader(testsetOut, batch_size=args.batchSize, shuffle=False, num_workers=2)
    cls_test_loader = torch.utils.data.DataLoader(cls_testset, batch_size=args.batchSize, shuffle=False, num_workers=2)


    num_class = int(max(testsetIn.targets) + 1)

    #Process
    for fold in range(1, args.max_fold+1):
        print("Processing fold {fold}".format(fold=fold))

        #load k th classifer
        if args.model == 'wide':
            net = wideresnet.WideResNet(int(num_class * 4 / 5))
        elif args.model =='dense':
            net = densenet.DenseNet(int(num_class * 4 / 5))
        elif args.model == 'resnet50':
            net = resnet.ResNet50(num_classes=num_class-1)

        net = nn.DataParallel(net).cuda()
        state_dict = torch.load(args.train_result_path+'/{model}_fold_{fold}/model_state_dict.pth'.format(model=args.model,fold=fold))
        net.load_state_dict(state_dict)
        net.eval()
        in_class = torch.load(args.train_result_path+'/{model}_fold_{fold}/in_class.pt'.format(model=args.model,fold=fold))

        testData(net, criterion, testloaderIn, testloaderOut, 'tomato_in', 'tomato_out', args.epsilon, args.temperature, fold, in_class,save_path,num_class)
        test_classifier(net,cls_test_loader,fold,in_class,save_path,num_class)
    #OOD detection
    ood_detection(len_out = len(testsetOut),in_dataset='tomato_in', out_dataset='tomato_out',save_path=save_path,testloaderIn=testloaderIn,cls_test_loader=cls_test_loader,num_class=num_class,testloaderOut=testloaderOut)
    print("Process Done")

    check_performance(save_path)


def testData(net,criterion, testloaderIn, testloaderOut, in_dataset,out_dataset, epsilon, temperature, fold, in_class,save_path,num_class):
    t0 = time.time()

    #nsplit = int(num_class * 0.8)
    nsplit = num_class-1

    in_sfx = np.array([])
    in_pro = np.array([])
    out_sfx = np.array([])
    out_pro = np.array([])
    ######################################In distribution data########################################
    for j, data in enumerate(testloaderIn):
        images,_ = data
        inputs = images.cuda().requires_grad_()
        outputs = net(inputs)

        #batch_size by num_class [0, 0 ... 0]
        o_output = np.zeros((images.size()[0], num_class))

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        #stack softmax output batch_size x num_class (for out_class, give 0)
        in_sfx = np.vstack((in_sfx, o_output)) if in_sfx.size else o_output

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
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)

        outputs = net(tempInputs)
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        # Stack Temperature scaled softmax probability on perturbed input

        in_pro = np.vstack((in_pro, o_output)) if in_pro.size else o_output
        if j % 100 == 99:
            print("{:4}/{:4} In Distribution images processed, {:.1f} seconds used.".format(j + 1, len(testloaderIn), time.time() - t0))
            t0 = time.time()

    ######################################OOD data########################################
    t0 = time.time()
    for j, data in enumerate(testloaderOut):
        images, _ = data

        inputs = images.cuda().requires_grad_()
        outputs = net(inputs)

        o_output = np.zeros((images.size()[0], num_class))
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]
        out_sfx = np.vstack((out_sfx, o_output)) if out_sfx.size else o_output

        # Using temperature scaling
        outputs = outputs / temperature

        o_output = np.zeros((images.size()[0], num_class))

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[:, 0] = (gradient[:, 0]) / 0.5
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)
        outputs = net(tempInputs)
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        # Stack Temperature scaled softmax probability on perturbed input

        out_pro = np.vstack((out_pro, o_output)) if out_pro.size else o_output
        if j % 100 == 99:
            print("{:4}/{:4} OOD images processed, {:.1f} seconds used.".format(j + 1, len(testloaderOut), time.time() - t0))
            t0 = time.time()
    data = {'in_sfx':in_sfx, 'in_pro':in_pro, 'out_sfx':out_sfx, 'out_pro':out_pro}
    pickle.dump(data, open(save_path+"/{in_dataset}_{out_dataset}_{fold}.p".format(in_dataset=in_dataset, out_dataset=out_dataset, fold=fold), "wb"))

def ood_detection(len_out,in_dataset, out_dataset,save_path,testloaderIn,num_class,cls_test_loader,testloaderOut):
    global ood_scores

    id = np.zeros(len(testloaderIn.dataset))
    ood = np.zeros(len_out)
    in_cls = np.zeros((len(testloaderIn.dataset),num_class))
    cls = np.zeros((len(cls_test_loader.dataset),num_class))



    for fold in range(1,args.max_fold+1):
        data = pickle.load(open(save_path+"/{in_dataset}_{out_dataset}_{fold}.p".format(in_dataset=in_dataset,out_dataset=out_dataset,fold=fold), "rb"))
        cls_data = pickle.load(open(save_path+"/classification_{fold}.p".format(fold=fold), "rb"))
        count = 0

        if len(data['in_pro']) <= len(data['out_pro']):
            range_max = len(data['out_pro'])
            range_min = len(data['in_pro'])


        for i in range(range_max):
            if i < range_min:
                # Get softmax
                in_probs = data['in_pro'][i]
                out_probs = data['out_pro'][i]
                # Get entropy
                in_probs_ = in_probs[np.nonzero(in_probs)]
                in_e = - np.sum(np.log(in_probs_) * in_probs_)
                out_probs_ = out_probs[np.nonzero(out_probs)]
                out_e = - np.sum(np.log(out_probs_) * out_probs_)
                # Get OOD score
                id[count] += (np.max(in_probs) - in_e)
                ood[count] += (np.max(out_probs) - out_e)
                # For baseline Scoring
                #id[count] += np.max(in_probs)
                #ood[count] += np.max(out_probs)

                count += 1
            else :
                out_probs = data['out_pro'][i]
                out_probs_ = out_probs[np.nonzero(out_probs)]
                out_e = - np.sum(np.log(out_probs_) * out_probs_)
                ood[count] += (np.max(out_probs) - out_e)

                count += 1

        in_cls += data['in_sfx']
        cls += cls_data['in_sfx']

    # Test on Classification
    pred = np.argmax(in_cls,axis=1)
    pred_new = np.argmax(cls,axis=1)
    target = testloaderIn.dataset.targets
    target_new = cls_test_loader.dataset.targets
    cls_err = 0.0
    cls_err_new = 0.0
    for i in range(len(target)):
        if pred[i] != target[i]:
            cls_err += 1
    cls_err = cls_err/len(target)
    for i in range(len(target_new)):
        if pred_new[i] != target_new[i]:
            cls_err_new += 1
    cls_err_new = cls_err_new/len(target_new)



    y_pred = id.tolist() + ood.tolist()
    ood_scores = sorted(y_pred)
    in_scores, out_scores = np.array(id), np.array(ood)
    target_in = np.array(testloaderIn.dataset.targets)
    target_out = np.array(testloaderOut.dataset.targets)

    fpr = metrics.tpr95(in_scores, out_scores,ood_scores)
    error,threshold = metrics.detection(in_scores, out_scores,ood_scores)
    auroc_ = metrics.auroc(in_scores, out_scores,ood_scores)
    auprin = metrics.auprIn(in_scores, out_scores,ood_scores)
    auprout = metrics.auprOut(in_scores, out_scores,ood_scores)

    # Confusion matrix
    y_pred = np.array(y_pred)
    y_pred = (y_pred>threshold)

    y_pred_in = y_pred[:len(id)]
    y_pred_out = y_pred[len(id):]

    in_confmat = metrics.confusion_matrix(y_pred_in,target_in,label=1)
    out_confmat = metrics.confusion_matrix(y_pred_out, target_out, label=0)
    confusion_matrix = np.concatenate((in_confmat,out_confmat),axis=0)


    performances = [fpr,error,auroc_,auprin,auprout,cls_err,cls_err_new]

    torch.save(performances,save_path+"/{model}_{in_dataset}_{out_dataset}_performance.p".format(model=args.model,in_dataset=in_dataset,out_dataset=out_dataset))
    np.savetxt(save_path+"/confusion_matrix.txt",confusion_matrix,fmt="%d")

def test_classifier(net, cls_test_loader, fold, in_class,save_path,num_class):
    t0 = time.time()

    nsplit = num_class - 1

    in_sfx = np.array([])
    for j, data in enumerate(cls_test_loader):
        images, _ = data
        inputs = images.cuda()
        outputs = net(inputs)

        # batch_size by 10 [0, 0 ... 0]
        o_output = np.zeros((images.size()[0], num_class))

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = nn.functional.softmax(outputs, dim=1)
        nnOutputs = nnOutputs.detach().cpu().numpy()
        for idx in range(nsplit):
            o_output[:, in_class[idx]] = nnOutputs[:, idx]

        # stack softmax output batch_size x num_class (for out_class, give 0)
        in_sfx = np.vstack((in_sfx, o_output)) if in_sfx.size else o_output

    data = {'in_sfx': in_sfx}
    pickle.dump(data, open(save_path + "/classification_{fold}.p".format(fold=fold), "wb"))
    print("Test classification task on Classifier : {0} Done".format(fold))


def check_performance(path):
    # out_dataset = ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize']
    # in_dataset = ['cifar10','cifar100']
    # model = ['wide','dense']
    #
    # for id in in_dataset:
    #     for m in model:
    #         for out in out_dataset:
    id = 'tomato_in'
    out = 'tomato_out'
    m = 'resnet50'
    print('in dataset : {id}   out dataset : {out}   model : {m}'.format(id=id, out=out, m=m))
    metric = torch.load(path+'/{m}_{id}_{out}_performance.p'.format(id=id, m=m, out=out))
    metric = [i * 100 for i in metric]
    print(metric)
    print('\n')



if __name__ == '__main__':
    main()


