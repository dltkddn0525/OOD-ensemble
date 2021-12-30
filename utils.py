from collections import Iterable
import numpy as np
import os
import torch
import shutil
import pickle
from matplotlib import pyplot as plt
import metrics



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log


def draw_curve(work_dir, train_logger, val_logger):
    train_logger = train_logger.read()
    val_logger = val_logger.read()

    epoch, train_loss = zip(*train_logger)
    epoch, val_auroc = zip(*val_logger)

    plt.plot(epoch,train_loss, '-b',label='Train Loss')
    #plt.plot(epoch,val_loss, '-r', label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.legend('upper right')
    plt.savefig(work_dir+'/loss_curve.png')
    plt.close()

    plt.plot(epoch,val_auroc,'r',label='Validation AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title("Validation AUROC")
    plt.legend('lower right')
    plt.savefig(work_dir+'/valauroc_curve.png')
    plt.close()

def ood_detection(len_in,len_out,temp,epsilon,num_class,save_path,max_fold):
    global ood_scores

    id = np.zeros(len_in)
    ood = np.zeros(len_out)

    for fold in range(1,max_fold+1):
        data = pickle.load(open(save_path+"/{temp}_{epsilon}_{fold}.p".format(temp=temp,epsilon=epsilon,fold=fold), "rb"))
        count = 0

        if len(data['in_pro']) < len(data['out_pro']):
            range_max = len(data['out_pro'])
            range_min = len(data['in_pro'])

        for i in range(range_max):
            if i < range_min:
                in_probs = data['in_pro'][i]
                out_probs = data['out_pro'][i]
                in_probs_ = in_probs[np.nonzero(in_probs)]
                in_e = - np.sum(np.log(in_probs_) * in_probs_)
                out_probs_ = out_probs[np.nonzero(out_probs)]
                out_e = - np.sum(np.log(out_probs_) * out_probs_)

                id[count] += (np.max(in_probs) - in_e)
                ood[count] += (np.max(out_probs) - out_e)

                count += 1
            else:
                out_probs = data['out_pro'][i]
                out_probs_ = out_probs[np.nonzero(out_probs)]
                out_e = - np.sum(np.log(out_probs_) * out_probs_)
                ood[count] += (np.max(out_probs) - out_e)

                count += 1


    ood_scores = id.tolist() + ood.tolist()
    ood_scores = sorted(ood_scores)
    id, ood = np.array(id), np.array(ood)

    fpr = metrics.tpr95(id, ood,ood_scores)
    error = metrics.detection(id, ood,ood_scores)
    auroc_ = metrics.auroc(id, ood,ood_scores)
    auprin = metrics.auprIn(id, ood,ood_scores)
    auprout = metrics.auprOut(id, ood,ood_scores)

    performances = [fpr,error,auroc_,auprin,auprout]

    return performances
