#biltsm+crf

这个demo主要是为了学习crf的实现

模型的结构如图：

<img src="/Users/fuhan/Library/Application Support/typora-user-images/image-20210607134127236.png" alt="image-20210607134127236" style="zoom:50%;" />

重点部分是CRF中如何手动构建特征，定义输入$X\in R^{n*d}$,n是输入词语长度，d是LSTM输出的特征的维度；输出$Y\in R^{n*d}$。以x1为例，是一个d维向量，每个元素对应一个特征。对于标注而言，特征就是标注的种类。

1）构建无向图模型中的团

$p(X,Y)=\frac{1}{Z}exp(\sum_i^T\sum_j^Kw_jF(X,Y_i))$，$Y_i$是团。这里y1,y2,X构成一个团，y2,y3,X构成一个团，...，yn-1,yn,X构成一个团

2）定义score函数，也就是F，分数越高概率越大

![image-20210607141059021](/Users/fuhan/Library/Application Support/typora-user-images/image-20210607141059021.png)

以团y0,y1,X为例子，y0为初始状态：

状态特征 --> y1的label对应的状态，找出x1中对应下标的值作为状态特征，x1[label_y1]

转移特征--> 定义一个(d+2)*(d+2)的转移矩阵T,增加的两个维度为句子的开始和结束状态,$T_{ij}$表示从j转化成i状态的score。两个状态关联越大，分数越高。

从代码看，$Score(y0,y1,X) = exp(T_{index(y1)index(y0)} + x_1[y_1])$

3）把所有的团的Score累乘，再归一化即可得到$p(X,Y)$

4）损失函数$L = -log(p(X,Y))$

5）利用SGD进行优化，每一个句子作为一个batch。

6）预测：由于此时不知道Y的标签，求解$argmax_{Y'}(p(X<Y'))$，$Y'$的可能总共有$d^n$个，可以用动态规划将时间复杂度降低到$o(n*d^2)$

