from matplotlib import cm
import os, gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./data/tomato_exp2/tomato_in/', type=str, help='directory of in distribution dataset')
parser.add_argument('--out_dir', default='./data/tomato_exp2/tomato_out/', type=str, help='directory of out of distribution dataset')
parser.add_argument('--save_path', default='./data/tomato_exp2_data/', type=str, help='save path')
parser.add_argument('--exp', default='exp1', type=str, help='type of experiment|exp1 or exp2')
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
save_path = args.save_path

if args.exp == 'exp1':
    test_count_in = 200
    test_count_out = 300
elif args.exp == 'exp2':
    test_count_in = 200
    test_count_out = 200

# Split in-dist data
trn_dir = save_path + 'trn_data/'
val_dir = save_path + 'val_data/'
cls_tst_dir = save_path + 'cls_tst_data/'
tst_dir = save_path + 'ood_tst_data/'

# Get Test Dataset
in_class_list = os.listdir(in_dir)
out_class_list = os.listdir(out_dir)

# Test in
for in_classes in in_class_list:
    src_path = os.path.join(in_dir,in_classes)
    file_list = os.listdir(src_path)
    np.random.shuffle(file_list)
    file_list = file_list[:test_count_in]

    dst_path = tst_dir+'in/'+in_classes
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for file in file_list:
        shutil.move(os.path.join(src_path,file),os.path.join(dst_path,file))

# Test out
for out_classes in out_class_list:
    src_path = os.path.join(out_dir,out_classes)
    file_list = os.listdir(src_path)
    np.random.shuffle(file_list)
    file_list = file_list[:test_count_out]

    dst_path = tst_dir+'out/'+out_classes
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for file in file_list:
        shutil.move(os.path.join(src_path,file),os.path.join(dst_path,file))


# Get Train & Validation Dataset
class_list = os.listdir(in_dir)
train_prop = 0.8
cls_tst_prop = 0.2
cls_tst_count = 0
val_count = 0

# Get In-distribution Train & classification test Dataset
for classes in class_list:
    file_list = os.listdir(os.path.join(in_dir,classes))
    np.random.shuffle(file_list)

    trn_file_list = file_list[:int(len(file_list)*train_prop)]
    cls_tst_file_list = file_list[int(len(file_list)*train_prop):]

    if not os.path.exists(trn_dir + classes):
        os.makedirs(trn_dir + classes)
    if not os.path.exists(cls_tst_dir + classes):
        os.makedirs(cls_tst_dir + classes)
    if not os.path.exists(val_dir + 'in/' + classes):
        os.makedirs(val_dir + 'in/' + classes)

    for file in trn_file_list:
        shutil.copy(os.path.join(in_dir+classes,file),trn_dir+classes)
    for file in cls_tst_file_list:
        shutil.copy(os.path.join(in_dir + classes, file),cls_tst_dir+classes)
        cls_tst_count +=1
        if cls_tst_count%2==0:
            shutil.copy(os.path.join(in_dir + classes, file), val_dir + 'in/' + classes)
            val_count+=1

# Get Out-of-distribution Validation Dataset
file_path = []
for folder in os.listdir(out_dir):
    path = os.path.join(out_dir,folder)
    for file in os.listdir(path):
        file_path.append(os.path.join(path,file))

np.random.shuffle(file_path)
val_ood_list = file_path[:val_count]
#if len(file_path)<(val_count+tst_count):
#    val_ood_list = file_path[:int(len(file_path)/2)]
#    tst_ood_list = file_path[int(len(file_path)/2):]
#else:
#    val_ood_list = file_path[:int(len(file_path)*0.1)]
#    tst_ood_list = file_path[int(len(file_path)*0.1):]
    # val_ood_list = file_path[:val_count]
    # tst_ood_list = file_path[val_count:val_count+tst_count]

if not os.path.exists(val_dir+'out/out'):
    os.makedirs(val_dir+'out/out')

for file in val_ood_list:
    shutil.copy(file,val_dir+'out/out')

# Print Data stat
train_data = 0
cls_tst_data = 0
val_data_in = 0
tst_data_in = 0
tst_data_out = 0
print("OOD Test Data-In")
for classes in sorted(os.listdir(tst_dir+'in/')):
    print("Class : {0}  Number : {1} ".format(classes,len(os.listdir(tst_dir+'in/'+classes))))
    tst_data_in +=len(os.listdir(tst_dir+'in/'+classes))
print("Total OOD Test data-In : {0}\n".format(tst_data_in))
print("OOD Test Data-Out")
for classes in sorted(os.listdir(tst_dir+'out/')):
    print("Class : {0}  Number : {1} ".format(classes,len(os.listdir(tst_dir+'out/'+classes))))
    tst_data_out +=len(os.listdir(tst_dir+'out/'+classes))
print("Total OOD Test data-In : {0}\n".format(tst_data_out))

print("In distribution Train Data")
for classes in sorted(os.listdir(trn_dir)):
    print("Class : {0}  Number : {1} ".format(classes,len(os.listdir(trn_dir+classes))))
    train_data += len(os.listdir(trn_dir+classes))
print("Total In Dist Train data : {0}\n".format(train_data))

print("In distribution Classification Test Data")
for classes in sorted(os.listdir(cls_tst_dir)):
    print("Class : {0}  Number : {1} ".format(classes,len(os.listdir(cls_tst_dir+classes))))
    cls_tst_data += len(os.listdir(cls_tst_dir+classes))
print("Total In Dist Train data : {0}\n".format(cls_tst_data))

print("In Distribution Validation Data")
for classes in sorted(os.listdir(val_dir+'in/')):
    print("Class : {0}  Number : {1} ".format(classes,len(os.listdir(val_dir+'in/'+classes))))
    val_data_in +=len(os.listdir(val_dir+'in/'+classes))
print("Total In Dist Validation data : {0}\n".format(val_data_in))

print("OOD Validation Data : {0}".format(len(os.listdir(val_dir+'out/out'))))
