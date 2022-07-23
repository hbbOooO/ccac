# CCAC 仓库

作者：韩明杰

创建时间：2022年7月17日

<hr>

## 仓库介绍
本仓库包括基于Bert、Roberta的对比学习框架，目前实现了在任务一上的搭建。

## 快速开始（以Roberta为例）
1. 训练对比学习模型。首先，打开文件`./configs/contrastive_roberta.yml`，修改数据文件目录`data_root_dir`，日志文件保存目录`log_dir`，修改权重文件保存目录`ckpt_dir`。运行 `ccac/hanmingjie/track1/run.py`,命令如下：
```bash
python run.py -config ./configs/contrastive_roberta.yml
```
2. 在训练集上测试模型。与步骤1类似，修改路径数据后，运行`ccac/hanmingjie/track1/run.py`,命令如下：
```bash
python run.py -config ./configs/contrastive_robertaclassification_head.yml
```
注：1）因为使用了两个不同的Model Class，所以该模型未实现边训边测的功能。目前设置每10个epoch保存一次权重，然后用分类头进行分类。

## 修改日志
| 时间 | 修改内容 |
| -- | -- |
| 7.17 | 创建基线 |
| 7.24 | 更新任务一的对比学习方法 |

## 目录建议
建议参与项目的同学隔离自己的代码，不要修改他人的代码。例如，张三同学可以将自己的所有代码放在 `ccac/zhangsan/` 目录下。

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



