# 基准数据集

AlphaMed 平台基准数据集均基于公开数据构建，不会泄露隐私数据。但是需要注意，使用时需要遵守数据集本身遵循的开源软件协议要求，以防范法律风险。

## 基准数据集功能简介

基准数据集用于帮助用户快速获取和处理公开数据集，使用户的精力集中在联邦学习任务本身，而不需要费心处理与原始数据相关的琐碎工作。为此，AlphaMed 平台上的所有基准数据集均支持以下特性：
- 具备自动下载数据的能力，免去人工寻找数据源、配置管理数据存储空间的麻烦；
- 具备数据集自动划分的能力，以方便在联邦任务的不同节点间部署不同数据，更精准的模拟联邦学习环境；
- 极简的数据加载方式，只需要在实例化时配置好必要的参数，即可得到可用的数据集对象，免去人工组织处理的繁琐操作；
- 方便使用，通过迭代器的方式按批次返回数据，一行代码融入训练流程；
- 支持数据预处理，方便根据具体任务和模型自定义数据和标签的输入形式；

在以下的基准数据集介绍中，可以直观的体会到以上这些特性带来的便利。

## FedMNIST 数据集

FedMNIST 是在 MNIST 数据集基础上增强实现的、用于联邦学习环境的数据集。FedMNIST 数据集的内容与 MNIST 数据集一致，但是预先将数据平均划分成了 10 等份，在加载数据集时可以通过指定 `client_ids` 参数加载对应的子集划分。在模拟联邦学习任务时，可以为不同的参与方指定加载不同的数据子集划分，以获得更加真实的实验效果。

需要注意的是，由于数据划分时采用的简单分配方式，FedMNIST 并不适合用于 Non-IID 场景下的算法效果验证，而是更适合于学习、演示，或者研究和工程的早起验证阶段。

### 加载 FedMNIST 数据集

加载 FedMNIST 的方法非常简单，与加载 MNIST 数据集很像。

```Python
import torchvision
from alphafed.dataset.mnist import FedMNIST

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
dataset = FedMNIST(root='DATA_DIR',
                   train=True,
                   download=True,
                   transform=transform,
                   client_ids=[3, 4])
```

上述代码加载 FedMNIST 数据集的训练数据的第 3 和第 4 份子集划分，如果本地没有数据，会从网络上自动下载。在训练过程中返回数据时，会先使用 `transform` 中定义的方法对数据做预处理。FedMNIST 数据集的初始化参数说明如下：
- root: 数据文件存放的根目录
- train: 加载训练集（True）还是测试集（False）
- download: 当本地不存在数据文件时，是否从网络上自动下载
- transform: 数据预处理函数，接受 `PIL.Image` 对象作为输入，比如 `transforms.RandomCrop`
- target_transform: 标签预处理函数
- client_ids: 指定加载的数据子集划分（传入单个 int 值）或子集划分列表（传入 int 值列表）。FedMNIST 将原始数据平均划分为 10 份，因此 `client_ids` 的取值范围为 [1, 10]

### 查看加载数据的摘要信息

只需直接打印数据集对象，就可以看到当前数据集中所加载的数据的摘要信息。

```Python
print(dataset)
```

```
# 期望输出：
Dataset FedMNIST
    Number of datapoints: 12000
    Root location: DATA_DIR
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.1307,), std=(0.3081,))
           )
    Features: [image, label]
    Available Client IDs: [1, 2, ..., 10]
    Current Client IDs: [3, 4]
```

### 集成 DataLoader 用于训练任务

你可以像使用任何 PyTorch Dataset 对象一样将 FedMNIST 传递给 DataLoader，然后在训练任务中使用。当然也可以直接使用 FedMNIST 数据集对象，取决于你的需求。以下是一个使用 DataLoader 的示例：

```Python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset=dataset,
                        shuffle=True,
                        batch_size=32)

for batch_data, batch_label in dataloader:
    # 正常训练逻辑
    ...
```
