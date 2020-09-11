# DeepHoughTransform
Add training codes into Hanqer/deep-hough-transform.

### Brief 
- Since there are no training codes in the original repo, I decided to add some training codes.
- I considered to use the architecture of deep-hough-transform to train on my own task. The task is not actually similar to the task of `Semantic Line Detection`.
- This repo is just an attempt of writing the training codes. No plan to open my dataset or evaluate the model.
- _(In Chinese):_ 这条用中文再啰嗦叙述一下，因为我个人目前的工作想要考虑看一下这个deep hough transform的方法能不能用在我的数据集上，但是他这个没有训练代码啊，自己试着写一下，不保证准确率那些。
- 多说一句，看代码的时候我发现trainset的设置（数据读取、transform）作者闭源了，另外label和loss计算也都闭源了，所以我考虑这两个点是有比较重点的tricks。

### Requirements
- Developed under `Pytorch1.3 Python3.6`


### Remarks
- (这段用中文记录一下复写的记录，后面可能会整理出一篇博客把作者的工作分析一下，另外补充一下代码实现细节)
- 截止目前为止(9/11/2020)，训练代码没有开源，除了网络结构之外，能够分析一下前后处理的就是forward.py脚本
```
key_points = model(images)
```
说明模型前向输出的是这个keypoints

### TODOs
- Not finished yet

