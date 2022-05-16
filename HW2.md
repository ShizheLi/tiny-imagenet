# DL HW2

---

### 一、计算 Resnet18 各层处理大小

- input：3 * 64 * 64

- conv1：64 * 32 * 32

- bn1：64 * 32 * 32

- relu：64 * 32 * 32

- maxpool：64 * 16 * 16

- layer1
  
  - BasicBlock0
    
    - conv1：64 * 16 * 16
    
    - bn1：64 * 16 * 16
    
    - relu：64 * 16 * 16
    
    - conv2：64 * 16 * 16
    
    - bn2：64 * 16 * 16
  
  - BasicBlock1
    
    - conv1：64 * 16 * 16
    
    - bn1：64 * 16 * 16
    
    - relu：64 * 16 * 16
    
    - conv2：64 * 16 * 16
    
    - bn2：64 * 16 * 16

- layer2
  
  - BasicBlock0
    
    - conv1：128 * 8 * 8
    
    - bn1：128 * 8 * 8
    
    - relu：128 * 8 * 8
    
    - conv2：128 * 8 * 8
    
    - bn2：128 * 8 * 8
    
    - donwsample：
  
  - BasicBlock1
    
    - conv1：128 * 8 * 8
    
    - bn1：128 * 8 * 8
    
    - relu：128 * 8 * 8
    
    - conv2：128 * 8 * 8
    
    - bn2：128 * 8 * 8

- layer3
  
  - BasicBlock0
    
    - conv1：256 * 4 * 4
    
    - bn1：256 * 4 * 4
    
    - relu：256 * 4 * 4
    
    - conv2：256 * 4 * 4
    
    - bn2：256 * 4 * 4
    
    - downsample：256 * 4 * 4
  
  - BasicBlock1
    
    - conv1：256 * 4 * 4
    
    - bn1：256 * 4 * 4
    
    - relu：256 * 4 * 4
    
    - conv2：256 * 4 * 4
    
    - bn2：256 * 4 * 4

- layer4
  
  - BasicBlock0
    
    - conv1：512 * 2 * 2
    
    - bn1：512 * 2 * 2
    
    - relu：512 * 2 * 2
    
    - conv2：512 * 2 * 2
    
    - bn2：512 * 2 * 2
    
    - downsample：512 * 2 * 2
  
  - BasicBlock1
    
    - conv1：512 * 2 * 2
    
    - bn1：512 * 2 * 2
    
    - relu：512 * 2 * 2
    
    - conv2：512 * 2 * 2
    
    - bn2：512 * 2 * 2

- avgpool：512 * 1 * 1

- fc：1000

- output：1000

![avatar](/Users/lishizhe/dl/models.png)

<br>

### 二、代码`main.py`的改写

- 导入`SummaryWriter`类

![avatar](/Users/lishizhe/Desktop/diff1.png)

- 修改网络输出类别，并修改对图片进行伸缩和裁剪的代码

![avatar](/Users/lishizhe/Desktop/diff2.png)

- 记录训练过程中的训练集和验证集的损失和精度变化

![avatar](/Users/lishizhe/Desktop/diff3.png)

- 修改对`函数train`和`validata`的定义，返回损失和精度

![avatar](/Users/lishizhe/Desktop/diff4.png)

<br>

### 三、验证数据集的代码编写

编写脚本将验证集的数据目录结构更改为与训练集一致，结果保存在`val.py`中，实现思路如下：

- 定义一个集合变量`val_dict`用来保存每张图片的以`.JPEG`结尾的名称和该图片的整数值标签

- 遍历所有图片，将图片名称和标签保存到`val_dict`中

- 再次遍历所有图片，以每张图片的标签为目录名创建文件夹，如果该文件夹存在就跳过

- 最后遍历一次所有图片，将每张图片按标签移动到对应文件夹中

- 删除原始文件夹

关键步骤实现如下：

```python
target_folder = './tiny-imagenet-200/val/'

val_dict = {}

with open('./tiny-imagenet-200/val/val_annotations.txt', mode='r') as f:

    for line in f.readlines():

        line = line.split('\t')

        val_dict[line[0]] = line[1]

paths = glob.glob('./tiny-imagenet-200/val/images/*')

for path in paths:

    file = path.split('/')[-1]

    folder = val_dict[file]

    if not os.exsists(target_folder + str(folder)):

        os.mkdir(target_folder + str(folder))

        os.mkdir(target_folder + str(folder) + '/images')

for path in paths:

    file = path.split('/')[-1]

    folder = val_dict[file]

    dest = target_folder + str(folder) + '/images/' + str(file)

    shutil.move(path, dest)

os.rmdir('./tiny-imagesnet-200/val/images')
```

<br>

### 四、训练

启动命令`python main.py -a resnet18 --epochs 25 --pretrained ./tiny-imagenet-200`，学习率 lr 、batch_size 、动量 momentum 等超参数保持默认。

训练结果曲线如下：

![avatar](/Users/lishizhe/dl/loss_accuracy.png)

由截图可以看出，随着 epoch 的迭代，训练集的损失在下降，而训练精度在上升。然而对于验证集而言，损失曲线和精度曲线都呈现了转折，即在 epoch = 7 之前，验证集损失在下降，且精度在上升，并在 epoch=7 达到极值点，这个趋势与训练集相同，但当 epoch 超过 7 之后，验证集的损失反而在上升，且精度在下降。因此，随着 epoch 的继续增加，网络在训练集上表现良好，但在验证集上的表现不佳，出现了过拟合现象。

<br>

### 五、模型 checkpoint 的保存和比较

保存模型的两个 checkpoint `checkpoint.pht.tar`和`model_best.pth.tar`，运行命令`python main.py --evaluate --resume ./checkpoint.pth.tar ./tiny-imagenet-200`和`python main.py --evaluate --resume ./model_best.pth.tar ./tiny-imagenet-200`，得到如下模型比较结果：

![avatar](/Users/lishizhe/Desktop/捕获.PNG)

可以看出，对于`checkpoint.pth.tar`，虽然训练的 epochs 更多，在训练集上的精度更高，但由于出现了过拟合现象，因此在测试集上的精度不高，平均只有68%左右。而对于`model_bets.pth.tar`，虽然只训练了10个 epochs，但在测试集上的精度接近最优，平均达到了72%左右。
