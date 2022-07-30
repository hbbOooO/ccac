# CCAC 智慧评测代码

作者：韩明杰

队伍名称：我真的栓Q队

创建时间：2022年7月17日

<hr>

## 比赛任务介绍

任务一：论点抽取及立场分类。

给定一个辩题和一个候选句子，用`<tab>`符号隔开， 参赛模型须判断当前句子是否为该辩题的论点， 并同时判断该论点（如有）的立场为支持或反对。 输出有三种标签：“1”表示该句子是论点且支持辩题，“-1”表示该句子是论点且反对辩题，“0”表示该句子不是论点。

任务二：论据发现

给定一个论点和一个候选句子，用<tab>符号隔开， 参赛模型须判断当前句子是否为支持当前论点的论据。 我们也会提供辩题作为辅助参考。输出有两种标签：1表示是论据，0表示非论据。

任务的更多细节内容见[CCAC 2022 第二届智慧论辩评测](http://www.fudan-disc.com/sharedtask/AIDebater22/index.html)。


## 比赛结果

| 任务名称 | 指标 | 名次 |
| -- | -- | -- |
| 任务一 | Macro F1=0.6442 | 2 |
| 任务二 | 论据上的F1=0.3538 | 3 |


## 仓库介绍
本仓库为本队伍实现的代码仓库，包括了数据、代码、训练结果。最终实现对[QNLI数据集](https://gluebenchmark.com/)的预训练，对任务一、任务二的分类，实现了[Stance Classification with Target-Specific Neural Attention Networks (TAN)](https://publications.aston.ac.uk/id/eprint/30835/1/Stance_Classification_with_Target_Specific_Neural_Attention_Networks.pdf)模型的复现。

支持使用Bert模型和Roberta模型，支持断点重新训练，目前只支持单卡训练。

## 分类结果
| 任务名称 | 是否使用预训练权重 | F1值（测试集） |
| -- | :--: | :--: |
| 任务一 | / | 0.5431 |
| 任务一 | 使用Loss=0.55的权重 | 0.6005 |
| 任务一 | 使用Loss=0.80的权重 | 0.5828 |
| 任务二 | / | 0.4220 |
| 任务二 | 使用Loss=0.55的权重 | 0.4469 |
| 任务二 | 使用Loss=0.80的权重 | 0.4530 |


## 快速开始

1. 任务一。首先，打开文件`ccac/hanmingjie/track1/configs/base/base.yml`，修改数据文件目录`data_root_dir`，日志文件保存目录`log_dir`，修改权重文件保存目录`ckpt_dir`。运行 `ccac/hanmingjie/track1/run.py`,命令如下：
```bash
python run.py -config ./configs/base/base.yml
```
在`base`目录下的`base_val.yml`用于在验证集上预测，`base_resume.yml`用于在保存的ckpt文件基础上继续训练，`base_pretrain.yml`用于在预训练的权重基础上训练。

2. 任务二。与任务一类似。

3. QNLI预训练。首先，打开文件`ccac/hanmingjie/track2/configs/qnli_roberta/finetune_roberta_qnli.yml`。修改相关路径的修改。运行 `ccac/hanmingjie/track2/run.py`,命令如下：
```bash
python run.py -config ./configs/qnli_roberta/finetune_roberta_qnli.yml
```

4. TAN模型。首先，打开文件`ccac/hanmingjie/track1/configs/tan/tan.yml`。修改相关路径的修改。运行 `ccac/hanmingjie/track2/run.py`,命令如下：
```bash
python run.py -config ./configs/tan/tan.yml
```


## 目录介绍
```
|--README.md
|--data    数据文件
|--save    权重目录
   |--qnli      在QNLI数据集上的预训练权重
      |--pretrain_loss_55    预训练时，Loss值为0.55时保存的权重
      |--pretrain_loss_80    预训练时，Loss值为0.80时保存的权重
   |--track1.baseline.model    任务一的训练结果
      |--train_from_none    从零开始训练结果
      |--train_from_pretrain_loss_55    从Loss=0.55的预训练的权重开始训练的结果
      |--train_from_pretrain_loss_80    从Loss=0.80的与预训练权重开始训练的结果
   |--track2.baseline.model    任务二的训练结果
      |--train_from_pretrain_loss_55    从Loss=0.55的预训练的权重开始训练的结果
      |--train_from_pretrain_loss_80    从Loss=0.80的与预训练权重开始训练的结果
|--hanmingjie    代码文件
   |--common    所有任务共用的代码
   |--submit    提交给CCAC的代码文件，能进行验证和测试，其中main.py是入口文件
   |--track1    任务一代码
      |--configs    配置文件
      |--datasets    数据集文件
      |--models    模型文件
      |--precessers    预处理文件
      |--util    工具文件
      |--run.py    入口文件
   |--track2    任务二代码（包括预训练任务），目录与任务一相同
```

## 修改日志
| 时间 | 修改内容 |
| -- | -- |
| 7.17 | 创建基线 |
| 7.24 | 更新任务一的对比学习方法 |
| 7.30 | 完成项目（删除对比学习方法） |

### 依赖
| 名称 | 版本 |
| --- | --- |
| python | 3.8 |
| pytorch | 1.8.1 |
| cuda | 11.1 |
| pytorch-transformers | 1.2.0 |
| scikit-learn | 1.1.1 |
| InfoNCE | 0.1.4 |
| GPU | TITAN XP(12G) |

Tips: TITAN XP 只能进行Bert训练，使用RoBerta需要更高版本的GPU，如2080Ti。


