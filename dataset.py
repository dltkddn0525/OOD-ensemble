from torchvision import transforms
import torch
import torchvision.datasets as datasets
import os
from PIL import Image
import numpy as np




def get_dataset(dataset_path,fold):

    # Set transform
    if dataset_path == './data/trn_data/CIFAR10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif dataset_path == './data/trn_data/CIFAR100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    else:
        mean = [0.5,0.5,0.5]
        stdv = [0.5,0.5,0.5]

    # Train mode
    if fold!=0:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize(64),
                                        transforms.RandomCrop(64,padding=8),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))
                                        ])

        in_dataset = datasets.ImageFolder(dataset_path, transform=transform)
        ood_dataset = datasets.ImageFolder(dataset_path, transform=transform)

        num_class = max(in_dataset.targets) + 1

        in_dataset, out_dataset, in_class = split_ood_train(in_dataset, ood_dataset, num_class, fold)

        return in_dataset, out_dataset, in_class, num_class

    # Validation or Test mode
    else:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))
                                        ])

        dataset = datasets.ImageFolder(dataset_path,transform=transform)

        return dataset

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, in_dataset, out_dataset,in_class=None):
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        self.in_class = in_class
        self.loader = pil_loader
        self.transform = self.in_dataset.transform

    def __getitem__(self, index):
        # If val_in data
        if index < len(self.in_dataset):
            path, target = self.in_dataset.samples[index]
            # If validation mode during training
            if self.in_class is not None:
                if target not in self.in_class:
                    target = -1
        # If val_out data
        else:
            path, target = self.out_dataset.samples[index-len(self.in_dataset)]
            target = -1

        img = self.loader(path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.in_dataset)+len(self.out_dataset)

def split_ood_train(in_dataset, out_dataset, num_classes,fold):
    np.random.seed(3)
    p1 = np.random.permutation(num_classes)
    split = int(num_classes/5)
    out_class = p1[(fold-1)*split:fold*split]
    in_class = []
    for i in p1:
        if i not in out_class:
            in_class.append(i)
    print("Fold : {0} \nOOD Class : {1} \nIn_Dist Class : {2}".format(fold,out_class,in_class))

    indata=[]
    outdata=[]
    in_label=[]
    out_label=[]

    for i in range(len(in_dataset)):
        path, target = in_dataset.samples[i]
        if target in in_class:
            temp = path,in_class.index(in_dataset.targets[i])
            indata.append(temp)
            in_label.append(in_class.index(in_dataset.targets[i]))
        else:
            temp = path,-1
            outdata.append(temp)
            out_label.append(-1)

    in_dataset.samples = indata
    out_dataset.samples = outdata
    in_dataset.targets = in_label
    out_dataset.targets = out_label

    return in_dataset, out_dataset, in_class

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
