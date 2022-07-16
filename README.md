# CCAC 仓库

作者：韩明杰

创建时间：2022年7月17日

<hr>

## 基线介绍
在track1任务下，用 Bert 实现的基本模型。训练1个epoch后，在验证集上 F1 指标为0.3212。其他任务track2、track3、track4暂未实现。

## 快速开始
运行 `ccac/hanmingjie/track1/run.py`,命令如下：
```bash
python run.py -config ./configs/base.yml
```

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
| GPU | TITAN XP(12G) |



