# Transformer Translation Model
此代码实现了论文[Attention is all you need]((https://arxiv.org/abs/1706.03762))中描述得Transformer translation model. 作者提供了tensorflow实现，这里我们将其转化为mxnet gluon模型
Transformer 是一种使用注意力机制(attention machanisms)的seq2seq模型，与传统seq2seq模型不同的是， Transformer不包含卷积神经网络(RNN)， 而是通过注意力机制来学习两个句子中tokens(这里可以不准确的理解为句子中的单词)
依赖关系， 并且这种注意力机制能够轻松的获取长距离依赖关系。
总的来说， Transformer还是尊崇了标准的encoder-decoder网络模式。 其中encoder使用自注意力机制(selfattention machanisms)来获取输入句子中包含的信息， decoder以encoder的输出和前面decoder的输出
作为输入， 每次产生一个token， 直至产生了终止符， 整个翻译过程结束。
与此同时， Transformer 使用了传统translation model中的embedding layer， 没有了卷积神经网路， 所以较之传统translation model， Transformer还多了位置编码(positional encoding)

## Contents
* [环境搭建]
* [程序流程]
* [程序概览]
* [术语定义]
* [文件位置]

## 环境搭建
python 2.7
mxnet 1.3.0
cuda 9.0
cudnn 7.1.3

## 程序流程
以下是运行程序流程：

### 下载测试集(计算BLEU)
使用以下命令
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de

### 下载训练集
我们使用了论文作者提供的训练集，这里附上下载链接

### 训练模型
使用以下命令：
python2 transformer_main.python2
默认的训练参数是：训练20轮(iteration)， 每一轮3200步(steps)。如果需要修改这两个训练参数， 请直接在transformer_main.py 的主函数中修改。其他参数在model_params.py文件中

### 查看训练结果并计算BLEU得分
训练模型参数保存在当前目录的‘transformer.params’中
训练BLEU结果保存在当前目录的‘blue_score_file’中

## 程序概览

### 模型定义
model文件夹中包含了Transformer模型的实现， 其中包含了以下文件：
* [transformer.py] 定义Transformer模型(主要包括了encoder/decoder层)
* [embedding_layer.py] embedding层， 这里embedding层的权重还用在计算decoder输出后， 计算pre-softmax probabilities, 这一点是为论文中没有体现， 但是作者提供的tensorflow代码中出现了。
* [attention_layer.py] 整个Transformer的重点， 也是核心部分， 定义了multi-head attention和 selfattention， 这在encoder/decoder层都有使用到
* [fnn_layer.py] 前向反馈传播层

其他文件还包括：
* [beam_search.py] 使用beam_search来寻找得分较高的几组翻译结果
* [model_params.py] 模型参数
* [model_utils.py] 定义一些辅助函数

### 模型训练/测试
* [transformer_main.py] 训练/测试过程
* [translate.py] 利用训练的网络进行翻译
* [compute_bleu.py] 翻译结果和翻译参照求bleu分数

其他文件还包括：
* [utils/dataset.py] 构造训练集， 利用get_mini_batch获取一个batch
* [utils/metrics.py] 定义了训练和评估中需要使用到的标准
* [utils/tokenizer.py] 产生‘字典’， 根据字典使模型输入数字化， 模型输出‘str'化

## 术语定义
对于code中一些变量名字， 以及注释中出现的术语， 以下是其解释(参照作者代码Readme)
**Steps / Epochs**:
* Step: unit for processing a single batch of data
* Epoch: a complete run through the dataset

Example: Consider a training a dataset with 100 examples that is divided into 20 batches with 5 examples per batch. A single training step trains the model on one batch.
 After 20 training steps, the model will have trained on every batch in the dataset, or one epoch.

**Subtoken**: Words are referred to as tokens, and parts of words are referred to as 'subtokens'. For example, the word 'inclined' may be split into `['incline', 'd_']`.
The '\_' indicates the end of the token. The subtoken vocabulary list is guaranteed to contain the alphabet (including numbers and special characters), so all words can be tokenized.

## 文件位置
全部文件都上传到github上面链接如下：
https://github.com/dong-jf15/TransformerTranslationModel.git
需要特别注意的是为了方便，我将训练用到的
'wmt32k-train.lang1'和‘wmt32k-trainl.lang2'
测试用到的
'newstest2014.en'和'newstest2014.de'
训练得到参数
'transformer.params'
都直接放在了transformer_main.py的同级目录下
