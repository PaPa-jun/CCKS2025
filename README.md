# CCKS2025 Competation Repo

## 任务简介

比赛任务是 NLP 中的经典任务：文本分类。具体来说，模型需要判断文本是由大模型生成的还是人类撰写的。

## 思路

目前的思路是利用预训练模型 Bert 在给定数据集上进行微调。由于训练集和测试集分布不太相同，因此我们需要尽可能的保证模型的泛化能力，避免过拟合。也就是尽可能捕捉数据集的有代表性的特征。

为了实现这个目标，我们考虑从两个角度入手：

- 传统机器学习的特征提取；
- DeTeCtive 对比学习框架。

### 传统机器学习特征提取

通过提取文本的各种特征，来丰富学习模型的学习材料，考虑：

- TF/IDF
- 文本丰富度

等。（这部分还需要完善）

### DeTeCtive 框架

DeTeCtive 是一种用于检测 AI 生成文本的新型框架，其核心思想是通过**多级对比学习**（Multi-Level Contrastive Learning）和**多任务辅助学习**（Multi-task Auxiliary Learning）来区分不同写作风格，而非传统的二分类（人类 vs. AI）。参考：[DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning](https://arxiv.org/abs/2410.20964)。

原论文中的 DeTeCtive 框架针对不同大模型生成的文本都做了不同力度的损失计算，而本次实验的数据集并不含具体的模型标签，仅有分类标签，因此我们不做不同粒度的损失计算，进考虑两部份损失：对比学习损失 + 分类损失。


## 代码架构

本仓库文件结构如下：
```
.
├── data                    # 数据集文件夹
│   ├── test.jsonl          # 测试集
│   └── train.jsonl         # 训练集
├── modules.py              # 类模块定义
├── predict.py              # 预测并生成提交文件
├── README.md               # README
├── train.py                # 训练代码
└── utils.py                # 工具函数
```

## 实验日志

- [6月3日] 从目前的实验结果来看，SimCES 框架效果并不好，直接微调 Bert 模型有过最好结果 83 分，另外使用 xgb 分类器效果也一般，下一步考虑使用微调 Bert + 集成学习分类器。
- [6月5日] 目前来看，Bert + 集成学习是一条可行的路子，将 Bert + xgb 的结果提升到了 75 分，特征方面，改进的 `get_features` 能达到 77 分。
- [6月6日] 直接调用 DeTeCtive 模型对 test 进行推理，F1 分数达到 80，下面考虑针对本次比赛数据集训练新的模型权重；参考 DeTeCtive 的正负样本对构造，重新实现对比学习框架并利用集成学习分类器成功将 F1 分数提升到 84.6 分！！！
