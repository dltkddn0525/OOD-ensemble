## Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers
- - -
### Introduction
1. In Distribution, Out of Distribution Data는 각각 아래와 같은 방식으로 준비되어야 합니다.
<pre>
In(out) data / class 0 / *.png
             / class 1 / *.png
                ...
             / class n / *.png
</pre>
2. data_setting.py 를 실행하여 데이터 셋을 나눕니다. Train 80% Validation 10% Test 10% 비율로 나눠지며, ood 데이터는 in 데이터의 validation set, test set의 수와 동일하게 무작위 추출됩니다.
<pre>
<code>
python data_setting.py --in_dir 'path to in distribution data folder' --out_dir 'path to ood data folder'
</code>
</pre>
Result
<pre>
data / trn_data / class 0 / *.png
                ...
                / class n / *.png
                
     / val_data / in / class 0 / *.png
                        ...
                     / class n / *.png
                / out / *.png
                
     / tst_data / in / class 0 / *.png
                        ...
                     / class n / *.png
                / out / *.png
</pre>
3. 원하는 beta, m 값을 적용하여 학습한 후(train.py), best performance를 보이는 beta, m으로 학습된 train_result path를 이용하여 최적의 T,epsilon을 찾습니다.(validation.py)

4. 찾아진 T, epsilon을 적용하여 ood detection을 수행합니다.(test.py)

### Training example
- Train WideResNet, CIFAR10 as in-dist, ensemble 5 classifier, with beta = 1, m = 0.4
<pre>
<code>
python train.py --train_dataset './data/trn_data/CIFAR10' --val_dataset './data/val_data/iSUN' 
                --save_path './train_result' --max_fold 5 --beta 1 --m 0.4
</code>
</pre>
- Result ({} = 각 fold에서의 best validation detection accuracy의 평균)
<pre>
train_result_{avg_val_performance} / wide_fold_1 / model_state_dict.pth
                                                   in_class.pth
                                                   loss_curve.png
                                                   train_logger.txt
                                                   val_logger.txt
                                   ...
                                   / wide_fold_5 / *
                                   configuration.json
</pre>

### Validation example
- Validation CIFAR10 as in-dist iSUN as out-dist, on DenseNet to search optimal hyperparameters(T, epsilon)
<pre>
<code>
python validation.py --in_dataset './data/tst_data/CIFAR10' --out_dataset './data/val_data/iSUN'
                     --train_result_path './train_result' --save_path './validation_result'
                     --model 'dense'
</code>
</pre>
- Result
<pre>
validation_result / validation_result_modelwide.csv
                    1_0_1.p # temperature_epsilon_fold.p, output of fold'th classifier
                    ...
                    5000_0.003_5.p
                    configuration.json
</pre>
### Test example
- Test CIFAR10 as in-dist Imagenet as out-dist on DenseNet with T=1000, epsilon=0.002
<pre>
<code>
python test.py --in_dataset './data/tst_data/CIFAR10' --out_dataset './data/tst_data/Imagenet'
               --train_result_path './train_result' --temperature 1000 --epsilon 0.002 
               --model 'dense' --save_path './test_result'
</code>
</pre>
- Result
<pre>
test_result / dense_CIFAR10_Imagenet_performance.p # FPR, Detection err, AUROC, AUPR_In, AUPR_Out, Cls_err
              CIFAR10_Imagenet_1.p # Output of fold'th classifier
              ...
              CIFAR10_Imagenet_5.p
              configuration.json
</pre>
### In-Distribution Datasets
- CIFAR10
- CIFAR100

### Out-of-Distribution Datasets
- TinyImageNet(Crop, Resize)
- LSUN(Crop, Resize)
- iSUN
- Gaussian Noise
- Uniform Noise

### Architecture
- DenseNet-BC (depth = 100, growth rate = 12, dropout rate = 0)
- WideResNet (depth = 28, width = 10, dropout rate = 0.3)

### Metric
- FPR at 95% TPR
- Detection Error
- AUROC
- AUPR(In, Out)

### Parameters
- Temperature(T)
- Magnitude(Epsilon)
- Number of Classifiers (K)
- Entropy Margin (m)
- Weight of Margin Entropy Loss (Beta)

### Searching Parameters (Paper)
- CIFAR-100 as In-distribution data and iSUN as out-of-distribution data on DenseNet-BC. Parameters which give best overall performance during validation were chosen.
- Temperature T = [1, 10, 100, 1000, 5000]
- Magnitude Epsilon = [0, 0.000313, 0.000625, 0.00125, 0.002, 0.003]

### Arguments for train.py
|Argument|Default|Decription|
|------|---|---|
|args.model|wide|model architecture (wide for WideResNet or dense for DenseNet-BC)|
|args.train_dataset|'./data/trn_data/CIFAR10'|path to training dataset : ./data/trn_data/CIFAR10 | ./data/trn_data/CIFAR100|
|args.val_dataset|'./data/val_data/iSUN'|path to validation dataset : ./data/val_data/iSUN|
|args.save_path|'./train_result'|result will be saved in this path|
|args.batchSize|100|Training batch size|
|args.max_fold|5|number of classifiers to ensemble|
|args.epoch|100|Training epoch|
|args.lr|0.1|Initial learning rate|
|args.final_lr|0.0001|Learning rate will be decayed until this value|
|args.momentum|0.9|momentum for SGD optimizer|
|args.weight_decay|5e-4|Weight decay|
|args.beta|0.2|coefficient of margin entropy loss|
|args.m|0.4|Least margin between avg entropy of id&ood|

### Arguments for validation.py
|Argument|Default|Decription|
|------|---|---|
|args.in_dataset|'./data/tst_data/CIFAR10'|path to validation in dataset : ./data/tst_data/CIFAR10 | ./data/tst_data/CIFAR100|
|args.out_dataset|'./data/val_data/iSUN'|path to validation ood dataset : ./data/val_data/iSUN|
|args.train_result_path|'./train_result'|path to train result directory : './train_result'|
|args.model|wide|model architecture (wide for WideResNet or dense for DenseNet-BC)|
|args.save_path|'./validation_result'|result will be saved in this path|

### Arguments for test.py
|Argument|Default|Decription|
|------|---|---|
|args.in_dataset|'./data/tst_data/CIFAR10'|path to test in dataset : ./data/tst_data/CIFAR10 | ./data/tst_data/CIFAR100|
|args.out_dataset|'./data/tst_data/Imagenet'|path to test ood dataset : ./data/tst_data/Imagenet|
|args.epsilon|0.002|Optimal input perturbation magnitude searched by validation|
|args.temperature|1000|Optimal scaling temperature searched by validation|
|args.model|wide|model architecture (wide for WideResNet or dense for DenseNet-BC)|
|args.save_path|'./validation_result'|result will be saved in this path|
|args.train_result_path|'./train_result'|path to train result directory : './train_result'|

### Training Setup
|Architecture|DenseNet-BC|WideResNet|
|------|---|---|
|Epochs|100|100|
|batch size|100|100|
|Loss|CrossEntropy + MarginEntropy|CrossEntropy + MarginEntropy|
|Optimizer|SGD(momentum=0.9)|SGD(momentum=0.9)|
|learning rate|0.1|0.1|
|weight decay|5e-4|5e-4|
|nesterov|Paper X/False|Paper X/False|
|learning rate decay|linearly drop to 0.0001|linearly drop to 0.0001|
|Number of Classifiers(K)|5|5|
|Margin m|0.4|0.4|
|Beta|Paper X/0.2|Paper X/0.2|


### Reference Code
- <https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers>
