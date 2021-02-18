# Introduction

## Supervised Learning(监督学习)

Supervised learning 需要大量的training data, 通过这个我们需要寻找一个function, 它的input和output之间的关系.

这种function的output, 通常被称作label(标签), 也就是说我们需要告诉机器, function的input和output分别是什么, 而这种output通常是通过人工的方式标注出来的, 因此被称为人工标注的label, 缺点是需要大量的人工effort.

### Regression(回归)

通过Regression找到的function, 它的输出是一个scalar数值

比如PM2.5的预测, 用过去的数据预测未来的数据, 就是一个典型的regression问题

### Classification(分类)

regression 和 classification 的区别是, 需要机器输出的类型是不一样的, 在regression里机器输出的是scalar, 而classification 分为两类:

#### Binary Classification(二元分类)

在这种情况下, 我们需要机器输出的是 Yes / No

比如Gmail 的spam filtering (垃圾邮件过滤器), 输入是邮件, 输出改邮件是否为垃圾邮件

#### Multi-class classification(多元分类)

在这种情况, 机器要做的是选择题, 等于给她数个选项, 一个选项一个类别, 他要从数个类别里面选择正确的类别

比如 document classification(新闻文章分类), 输入一则新闻, 输出这个新闻属于哪一类

### model(function set) 选择模型

在解任务的过程种, 第一步是要选一个function的set, 选不同function的set会得到不同的结果;
而选不同的function set就是选不同的model, model又分为很多种:

- Linear Model(线性模型): 最简单的模型
- Non-linear Model(非线性模型): 最常用的模型:
    - deep learning: alpha-go
    - SVM
    - Decision Tree
    - K-NN

## Semi-supervised Learning(半监督学习)

如果想要做一个区分猫和狗的function

手头上有少量的labeled data, 他们标注了图片上哪只是猫哪只是狗; 同时又有大量的unlabeled data, 只有图片没有标注去告诉机器哪只是猫和狗

在Semi-supervised Learning的技术里, 这些没有labeled的data, 对机器学习也是有帮助的

## Transfer Learning(迁移学习)

猫狗分类的问题, 只有少量labeled data, 但是现在有大量的不相干的data(不是猫和狗的图片), 在这些大量的data里面, 他可能有label也可能没有label

Transfer Learning要解决的问题是, 这一堆不相干的data可以对结果带来什么帮助

## Unsupervised Learning(无监督学习)

无师自通. 如果我们给机器看大量的文章, 他能狗做什么事情?

又比如，我们带机器去逛动物园，给他看大量的动物的图片，对于unsupervised learning来说，我们的data中只有给function的输入的大量图片，没有任何的输出标注；在这种情况下，机器该怎么学会根据testing data的输入来自己生成新的图片

## Structured Learning(结构化学习)

机器要输出一个有结构的东西

在分类的问题中, 机器输出的只是一个选项; 在structured类的problem里面, 机器要输出的是一个复杂的物件

举例来说, 在语音识别的情境下, 机器输出的是一个声音信号, 输出的是一个句子; 句子是由许多词汇拼凑而成, 他是一个有结构性的object
或者说机器翻译, 人脸识别

## Reinforcement Learning(强化学习)

### Supervised Learning

会告诉机器正确答案, 其特点是 **Learning from teacher**

- 训练一个聊天机器人, 告诉他如果使用者说了“Hello”，你就说“Hi”；如果使用者说了“Bye bye”，你就说“Good bye”；就好像有一个家教在它的旁边手把手地教他每一件事情

### Reinforcement Learning

我们没有告诉机器正确的答案是什么，机器最终得到的只有一个分数，就是它做的好还是不好，但他不知道自己到底哪里做的不好，他也没有正确的答案；很像真实社会中的学习，你没有一个正确的答案，你只知道自己是做得好还是不好。其特点是**Learning from critics**

- 训练一个聊天机器人，让它跟客人直接对话；如果客人勃然大怒把电话挂掉了，那机器就学到一件事情，刚才做错了，它不知道自己哪里做错了，必须自己回去反省检讨到底要如何改进，比如一开始不应该打招呼吗？还是中间不能骂脏话之类的

再拿下棋这件事举例，supervised Learning是说看到眼前这个棋盘，告诉机器下一步要走什么位置；而reinforcement Learning是说让机器和对手互弈，下了好几手之后赢了，机器就知道这一局棋下的不错，但是到底哪一步是赢的关键，机器是不知道的，他只知道自己是赢了还是输了

其实Alpha Go是用supervised Learning+reinforcement Learning的方式去学习的，机器先是从棋谱学习，有棋谱就可以做supervised的学习；之后再做reinforcement Learning，机器的对手是另外一台机器，Alpha Go就是和自己下棋，然后不断的进步