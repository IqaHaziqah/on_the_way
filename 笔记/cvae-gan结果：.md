---
typora-root-url: image
---

```
v6用于基本参数，v7用于架构修改
```

> https://blog.csdn.net/u012436149/article/details/70264257 用于scope下的正则化
>

#### 理论指导：

![1526285160294](/1526285160294.png)

##### GAN training tricks

##### 不能在一次梯度中，既使用生成样本，又使用真实样本，这样会使discriminator很容易获得两个样本间的差异，而使模型坍塌？

feature matching

现实的理论依据是，生成器不仅要使得分辨器生成错误，还要生成样本和真实样本的统计特征相似；特别是真实样本在分辨器中的特征

minibatch discrimination #论文中说明该方案可以很快训练出能力强的生成器 vote 2

当生成器坍缩时，生成样本会非常相似，生成器只能生成固定的样本来通过分辨器的测试，在这个问题上，有效的方案是增加生成样本间的相似性，并惩罚大的相似性。

从目标函数的角度对相似样本进行惩罚，

historical averaging

这个故事的意思是，使用历史参数的数据，计算每个参数同其历史平均的差值，并将其作为一个正则项。

batch normalization

为了防止少量的样本偏移问题，使用BN，为了使x和其他x‘独立，则采用VBN，即使用固定的参数对样本进行重新计算，有一个reference batch是固定的，



single-sided smoothing 

将正类样本的label从1改成0.9，这部分应该有理论依据。

https://ctmakro.github.io/site/on_learning/gan.html 单边平滑和minibatch discrimination

在check gradients里，有理论计算梯度和数值计算梯度，理论计算即为根据公式推导出$f'(x)$ 后计算梯度，数值梯度是根据导数定义计算得来的。

[1]: http://jacoxu.com/jacobian%E7%9F%A9%E9%98%B5%E5%92%8Chessian%E7%9F%A9%E9%98%B5/	"Jacobian矩阵和Hessian矩阵"

[2]: https://sinpycn.github.io/2017/05/09/GAN-Tutorial-Tips-and-Tricks.html	"GAN介绍-提示与技巧"

loss函数：因为opt = optimizer.minize(loss)，所以loss最终应该是趋近于0，且从现实角度看应该是越小越好。

cross_entropy: $cross_entropy = -\sum y_{true}*logy_{pred}$ 

之所以使用logvar是因为该变量的范围不受约束，可以直接从线性网络输出。

GAN的generator减弱模型坍塌问题：增强模型输出的多样性

> 1. “用上几轮训练的loss,而不是本轮的loss 来更新判别器”（Learning from Simulated and Unsupervised Images through Adversarial Training）；Learning from Simulated and Unsupervised Images through Adversarial Training
> 2. 在判别器网络的最后加一个“minibatch layer”，用一个大的张量来累计统计该batch中所有样本的，并作为输出。1710.10196.pdf
> 3. 将判别器D展开（unrolled）并回滚 1611.02163.pdf

多样性度量标准：

> 首次看到了他们用multi-scale structural similarity (MS-SSIM)这种指标，来衡量“一批量图片之间的相似度”，指标分数越高，一批图的相似度就越高。之后看到的几篇工作里也引用了这种做法。1610.09585.pdf

cvae-gan结果：

一次10折交叉验证



#### check plan

确保代码能正确运行：

check all the loss and the network（已修正目前所发现问题，4.19）

check the gradients (已随机检查过梯度，但是encoder的部分梯度差异过大，decoder部分则暂时还未检查)

修改整体network精度后，相对误差恢复至正常范围，使用双精度浮点数，随机检查没有问题

代码能收敛：

check the whole loss

discriminator收敛很快，但generator未收敛，其余在正常范围内，但现在使用的是minibatch下的loss函数，并未使用整体数据集的loss函数，classifier在最初的几个batch后loss值也基本不再变化

生成效果不佳：

check the generation

##### questions：

不好计算loss

KNN的计算结果一致，证明生成样本范围有限，根据5—NN结果显示，在产生200个样本的情况下，只有3个正类样本的近邻，数量超过140个，且还会生成负类样本周围的新样本。

不符合over-sampling的基本法则，即采样率越高少数类准确率越高，这部分考虑生成样本范围，以及隐变量是否真实符合正态分布。

encoder和genenrator的收敛很慢，同discriminator及classifier不是一个水平的，使用差异化的学习率确实能均匀其网络收敛速度。



#### 理论分析

首先：以下分析建立在该假设上：即生成样本非常相似，变化范围很小，包含信息有限，更多的生成样本不会导致分类器效果变好，反倒因为增加了噪声而效果变差。

原因分析：discriminator 导致了generator的收缩，即生成其他样本均会被dis识别，而只能生成固定的样本，在这个过程中，（分析验证：dis产生的误差要远大于平方误差），生成样本种类有限。

削弱dis的分类作用，保证KL和重建误差在generator中的意义，否则会降低generator的生成效果。

此时，可以考虑在生成后加入样本筛选，但至少要保证，生成的样本是有意义的，否则筛选失去作用。

#### 梯度检验

theory

在神经网络的实现过程中，对网络中计算出的梯度和自己计算的数值梯度进行比较，主要是根据梯度的定义来计算，即

注意：是对求导的自变量进行变换，且在计算偏导时，一次只能对一个自变量的分量进行变换。

实际应用中，我们常将$\epsilon$ 设为一个很小的常量，比如在$10^{-4}$ 数量级（虽然$\epsilon$ 取值范围可以很大，但是我们不会将它设得太小，比如$10^{-20}$ 因为那将导致数值舍入误差。）

code

tensorflow中有实现这两部分检验的函数：tf.test.compute_gradient_error 和 tf.test.compute_gradient，前者返回两种梯度间的误差，而后者返回两种计算方式下的梯度。

#### IDEA

样本筛选：如果生成样本的K-近邻中多数类大于少数类，则会增加原始样本的IGIR，由于IGIR不具有对称性，所以该想法排除；

对于原始样本来说，如果

#### ionosphere 

```
para_o = {
    'latent_dim':2,
    'lam':0,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.0002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.685433056795 mean f-maj: 0.890892424903 mean accuracy: 0.83850140056
mean gmean: 0.727943893496 mean TPrate: 0.551923076923 mean AUC: 0.943720229046
=====NB+vae=====
mean F1-min: 0.66504182451 mean f-maj: 0.882616708363 mean accuracy: 0.826732026144
mean gmean: 0.709209325381 mean TPrate: 0.517948717949 mean AUC: 0.917915779872
=====NB+vae=====
mean F1-min: 0.624541062802 mean f-maj: 0.874525947763 mean accuracy: 0.812759103641
mean gmean: 0.676703366038 mean TPrate: 0.478846153846 mean AUC: 0.86864548495
time used: 677.5170815645424
```

```
para_o = {
    'latent_dim':2,
    'lam':0,
    'epochs':1,
    'batch_size':20, #为偶数
    'learning_rate':0.0002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.674702062643 mean f-maj: 0.886357721312 mean accuracy: 0.832455648926
mean gmean: 0.719669556001 mean TPrate: 0.542948717949 mean AUC: 0.942091061113
=====NB+vae=====
mean F1-min: 0.64829636237 mean f-maj: 0.882125110334 mean accuracy: 0.824206349206
mean gmean: 0.69784217374 mean TPrate: 0.512179487179 mean AUC: 0.957465541705
=====NB+vae=====
mean F1-min: 0.627904205507 mean f-maj: 0.879190501083 mean accuracy: 0.818650793651
mean gmean: 0.681595381145 mean TPrate: 0.496794871795 mean AUC: 0.958283926219
time used: 87.6636644324335
```

```
para_o = {
    'latent_dim':2,
    'lam':0,
    'epochs':100,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.698999669211 mean f-maj: 0.894677153957 mean accuracy: 0.844383753501
mean gmean: 0.739214206872 mean TPrate: 0.56858974359 mean AUC: 0.944742576264
=====NB+vae=====
mean F1-min: 0.665955083788 mean f-maj: 0.882840552649 mean accuracy: 0.826900093371
mean gmean: 0.712220463856 mean TPrate: 0.527564102564 mean AUC: 0.920903010033
=====NB+vae=====
mean F1-min: 0.620337416431 mean f-maj: 0.872665819688 mean accuracy: 0.810070028011
mean gmean: 0.675036003395 mean TPrate: 0.480769230769 mean AUC: 0.882286915983
time used: 142.24209962791974  
```

```
para_o = {
    'latent_dim':2,
    'lam':0,
    'epochs':100,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':None,#sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.685433056795 mean f-maj: 0.890892424903 mean accuracy: 0.83850140056
mean gmean: 0.727943893496 mean TPrate: 0.551923076923 mean AUC: 0.945239181109
=====NB+vae=====
mean F1-min: 0.656968119749 mean f-maj: 0.881154237098 mean accuracy: 0.8239589169
mean gmean: 0.703317070499 mean TPrate: 0.510897435897 mean AUC: 0.921280531063
=====NB+vae=====
mean F1-min: 0.620580180203 mean f-maj: 0.872606697262 mean accuracy: 0.809743230626
mean gmean: 0.674187500415 mean TPrate: 0.471153846154 mean AUC: 0.881175889328
time used: 142.47117560223523
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.702099723462 mean f-maj: 0.894277040288 mean accuracy: 0.844383753501
mean gmean: 0.740919544696 mean TPrate: 0.56858974359 mean AUC: 0.944318181818
=====NB+vae=====
mean F1-min: 0.708629416679 mean f-maj: 0.893548547313 mean accuracy: 0.844379084967
mean gmean: 0.744640334466 mean TPrate: 0.567948717949 mean AUC: 0.912796442688
=====NB+vae=====
mean F1-min: 0.663909160155 mean f-maj: 0.88366736007 mean accuracy: 0.827549019608
mean gmean: 0.709330826029 mean TPrate: 0.521153846154 mean AUC: 0.864169707104
time used: 680.8830808010462        
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[10,10,10,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.685433056795 mean f-maj: 0.890892424903 mean accuracy: 0.83850140056
mean gmean: 0.727943893496 mean TPrate: 0.551923076923 mean AUC: 0.944756511604
=====NB+vae=====
mean F1-min: 0.67973787837 mean f-maj: 0.886424799326 mean accuracy: 0.832782446312
mean gmean: 0.721434972909 mean TPrate: 0.535897435897 mean AUC: 0.908242120199
=====NB+vae=====
mean F1-min: 0.60862150921 mean f-maj: 0.871322652444 mean accuracy: 0.807296918768
mean gmean: 0.663579739047 mean TPrate: 0.465384615385 mean AUC: 0.855975727171
time used: 691.3180911475538        
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[30,10,30,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.702099723462 mean f-maj: 0.894277040288 mean accuracy: 0.844383753501
mean gmean: 0.740919544696 mean TPrate: 0.56858974359 mean AUC: 0.943744299179
=====NB+vae=====
mean F1-min: 0.719245082403 mean f-maj: 0.89467198071 mean accuracy: 0.847072829132
mean gmean: 0.754320507369 mean TPrate: 0.583333333333 mean AUC: 0.91800319246
=====NB+vae=====
mean F1-min: 0.672694289005 mean f-maj: 0.882389624734 mean accuracy: 0.827301587302
mean gmean: 0.716464326759 mean TPrate: 0.528205128205 mean AUC: 0.867991790818
time used: 702.1235199090352
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[30,10,30,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.70109869147 mean f-maj: 0.894220635723 mean accuracy: 0.844220354809
mean gmean: 0.740201905536 mean TPrate: 0.567948717949 mean AUC: 0.942367234215
=====NB+vae=====
mean F1-min: 0.672720334511 mean f-maj: 0.884628880959 mean accuracy: 0.829841269841
mean gmean: 0.715769389446 mean TPrate: 0.527564102564 mean AUC: 0.901874936658
=====NB+vae=====
mean F1-min: 0.637865575257 mean f-maj: 0.875748703619 mean accuracy: 0.815546218487
mean gmean: 0.68727037176 mean TPrate: 0.487820512821 mean AUC: 0.851830596939
time used: 702.3878624758545        
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[30,10,30,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.002,0.002],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.694256586207 mean f-maj: 0.892551550092 mean accuracy: 0.841442577031
mean gmean: 0.734758589014 mean TPrate: 0.560256410256 mean AUC: 0.941128255802
=====NB+vae=====
mean F1-min: 0.707700534759 mean f-maj: 0.893419902596 mean accuracy: 0.844131652661
mean gmean: 0.743883812407 mean TPrate: 0.566666666667 mean AUC: 0.904830495591
=====NB+vae=====
mean F1-min: 0.667405945597 mean f-maj: 0.882753150736 mean accuracy: 0.827138188609
mean gmean: 0.710800406537 mean TPrate: 0.519230769231 mean AUC: 0.864686581534
time used: 694.0027340825181
```

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.002,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[30,10,30,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.682573800618 mean f-maj: 0.891410761123 mean accuracy: 0.838664799253
mean gmean: 0.726480256601 mean TPrate: 0.552564102564 mean AUC: 0.943297101449
=====NB+vae=====
mean F1-min: 0.713015873016 mean f-maj: 0.895388598283 mean accuracy: 0.847072829132
mean gmean: 0.748608229746 mean TPrate: 0.575 mean AUC: 0.900043072869
=====NB+vae=====
mean F1-min: 0.675170770062 mean f-maj: 0.886968886143 mean accuracy: 0.832857142857
mean gmean: 0.718542626638 mean TPrate: 0.535256410256 mean AUC: 0.85391076315
time used: 3105.9270292831497        
```

```
WARNING:root:the para_o is {'learning_rate': 0.005, 'lamda': [2.5, 1, 1, 1, 0.1], 'latent_dim': 2, 'batch_size': 20, 'dataset_name': 'ionosphere', 'PRE': <class 'sklearn.preprocessing.data.StandardScaler'>, 'input_dim': 34, 'epochs': 1000, 'hidden': [20, 10, 20, 10]}
WARNING:root:the answer is ([0.67855385285658554, 0.72352447025395117, 0.55128205128205132], [0.65828808633223768, 0.70534777005998728, 0.51923076923076927], [0.63391351353661218, 0.68512648038920732, 0.48782051282051286])    
```

##### 改变learning_rate

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.01,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[30,10,30,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
#此处修改的是learning_rate        
0.005
=====NB+vae=====
mean F1-min: 0.702494980379 mean f-maj: 0.894180421931 mean accuracy: 0.844383753501
mean gmean: 0.74111787333 mean TPrate: 0.56858974359 mean AUC: 0.9477652782
=====NB+vae=====
mean F1-min: 0.6558403002 mean f-maj: 0.881435506076 mean accuracy: 0.824122315593
mean gmean: 0.702826636262 mean TPrate: 0.511538461538 mean AUC: 0.916768014594
=====NB+vae=====
mean F1-min: 0.619064872326 mean f-maj: 0.872779588662 mean accuracy: 0.809827264239
mean gmean: 0.672978635126 mean TPrate: 0.471794871795 mean AUC: 0.853645991689

0.001
=====NB+vae=====
mean F1-min: 0.698999669211 mean f-maj: 0.894677153957 mean accuracy: 0.844383753501
mean gmean: 0.739214206872 mean TPrate: 0.56858974359 mean AUC: 0.94175914665
=====NB+vae=====
mean F1-min: 0.699571879789 mean f-maj: 0.891957893054 mean accuracy: 0.841437908497
mean gmean: 0.737794637641 mean TPrate: 0.559615384615 mean AUC: 0.90536257221
=====NB+vae=====
mean F1-min: 0.663148926237 mean f-maj: 0.883729178057 mean accuracy: 0.827549019608
mean gmean: 0.708854887796 mean TPrate: 0.521153846154 mean AUC: 0.861268622682

0.0005
=====NB+vae=====
mean F1-min: 0.672573800618 mean f-maj: 0.889814244432 mean accuracy: 0.835723622782
mean gmean: 0.718745229682 mean TPrate: 0.544230769231 mean AUC: 0.942414107631
=====NB+vae=====
mean F1-min: 0.70049772381 mean f-maj: 0.89375653704 mean accuracy: 0.843884220355
mean gmean: 0.738991182707 mean TPrate: 0.565384615385 mean AUC: 0.909184655924
=====NB+vae=====
mean F1-min: 0.683203896356 mean f-maj: 0.888277122258 mean accuracy: 0.835550887021
mean gmean: 0.724418368384 mean TPrate: 0.542307692308 mean AUC: 0.862506334245
```

##### 改变hidden

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.0005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[10,5,10,5],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
#改变hidden
[10, 10, 10, 10]
=====NB+vae=====
mean F1-min: 0.686654247026 mean f-maj: 0.890944282447 mean accuracy: 0.838664799253
mean gmean: 0.72883933426 mean TPrate: 0.552564102564 mean AUC: 0.946051231377
=====NB+vae=====
mean F1-min: 0.665791649161 mean f-maj: 0.882813362286 mean accuracy: 0.826900093371
mean gmean: 0.710131766016 mean TPrate: 0.519230769231 mean AUC: 0.91108999696
=====NB+vae=====
mean F1-min: 0.622522438002 mean f-maj: 0.872343939333 mean accuracy: 0.809906629318
mean gmean: 0.67433086819 mean TPrate: 0.471794871795 mean AUC: 0.858371338806
[20, 10, 20, 10]
=====NB+vae=====
mean F1-min: 0.698999669211 mean f-maj: 0.894677153957 mean accuracy: 0.844383753501
mean gmean: 0.739214206872 mean TPrate: 0.56858974359 mean AUC: 0.943330039526
=====NB+vae=====
mean F1-min: 0.662479147959 mean f-maj: 0.885847248235 mean accuracy: 0.830168067227
mean gmean: 0.708760728396 mean TPrate: 0.528846153846 mean AUC: 0.904981250633
=====NB+vae=====
mean F1-min: 0.615586814436 mean f-maj: 0.875379845777 mean accuracy: 0.813333333333
mean gmean: 0.651606270145 mean TPrate: 0.48141025641 mean AUC: 0.848027515962
```

##### 改变初始化方式

开始为高斯分布，后改为均匀分布

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.0005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[10,10,10,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.718198794669 mean f-maj: 0.897433129482 mean accuracy: 0.849939309057
mean gmean: 0.753386147122 mean TPrate: 0.583974358974 mean AUC: 0.946093037397
=====NB+vae=====
mean F1-min: 0.736197266786 mean f-maj: 0.900177643707 mean accuracy: 0.855406162465
mean gmean: 0.766591189633 mean TPrate: 0.598076923077 mean AUC: 0.911139404074
=====NB+vae=====
mean F1-min: 0.669120157598 mean f-maj: 0.882938644342 mean accuracy: 0.827385620915
mean gmean: 0.712505110285 mean TPrate: 0.520512820513 mean AUC: 0.855318232492
```

```
修改初始化方式后，对learning_rate进行修改：
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.0005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[10,10,10,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
for value in [0.01,0.005,0.001,0.0005,0.0002]:
0.01
=====NB+vae=====
mean F1-min: 0.737148624098 mean f-maj: 0.902829803036 mean accuracy: 0.858510737628
mean gmean: 0.768990024641 mean TPrate: 0.607051282051 mean AUC: 0.946791071248
=====NB+vae=====
mean F1-min: 0.729090864416 mean f-maj: 0.898583326457 mean accuracy: 0.852871148459
mean gmean: 0.762853574874 mean TPrate: 0.598717948718 mean AUC: 0.905969392926
=====NB+vae=====
mean F1-min: 0.701603539967 mean f-maj: 0.891685740664 mean accuracy: 0.841596638655
mean gmean: 0.740711733599 mean TPrate: 0.567307692308 mean AUC: 0.853075909598
=====C4.5+vae=====
mean F1-min: 0.82812701791 mean f-maj: 0.908027291549 mean accuracy: 0.880980392157
mean gmean: 0.861886511223 mean TPrate: 0.819871794872 mean AUC: 0.867939850005
=====C4.5+vae=====
mean F1-min: 0.824566230871 mean f-maj: 0.90287480151 mean accuracy: 0.875499533147
mean gmean: 0.859682876877 mean TPrate: 0.820512820513 mean AUC: 0.863813722509
=====C4.5+vae=====
mean F1-min: 0.82442654214 mean f-maj: 0.903505726823 mean accuracy: 0.875994397759
mean gmean: 0.861699100735 mean TPrate: 0.829487179487 mean AUC: 0.866225803182
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206

0.005
=====NB+vae=====
mean F1-min: 0.779675869516 mean f-maj: 0.913090592599 mean accuracy: 0.875504201681
mean gmean: 0.802897483721 mean TPrate: 0.654487179487 mean AUC: 0.941182730313
=====NB+vae=====
mean F1-min: 0.753816314405 mean f-maj: 0.905874271294 mean accuracy: 0.864145658263
mean gmean: 0.781563469689 mean TPrate: 0.622435897436 mean AUC: 0.90478995642
=====NB+vae=====
mean F1-min: 0.712950513539 mean f-maj: 0.892989506926 mean accuracy: 0.844374416433
mean gmean: 0.749012524119 mean TPrate: 0.575 mean AUC: 0.852594506942
=====C4.5+vae=====
mean F1-min: 0.852005844397 mean f-maj: 0.921864458171 mean accuracy: 0.897969187675
mean gmean: 0.878989429775 mean TPrate: 0.827564102564 mean AUC: 0.882754383298
=====C4.5+vae=====
mean F1-min: 0.800737885303 mean f-maj: 0.890254974751 mean accuracy: 0.858748832866
mean gmean: 0.838271737537 mean TPrate: 0.780128205128 mean AUC: 0.841645130232
=====C4.5+vae=====
mean F1-min: 0.843381430164 mean f-maj: 0.911488862986 mean accuracy: 0.887100840336
mean gmean: 0.875857933715 mean TPrate: 0.84358974359 mean AUC: 0.87772372555
=====3-NN+vae=====
mean F1-min: 0.737754019436 mean f-maj: 0.892114523254 mean accuracy: 0.847399626517
mean gmean: 0.773693986213 mean TPrate: 0.623717948718 mean AUC: 0.900381955002
=====3-NN+vae=====
mean F1-min: 0.737754019436 mean f-maj: 0.892114523254 mean accuracy: 0.847399626517
mean gmean: 0.773693986213 mean TPrate: 0.623717948718 mean AUC: 0.900381955002
=====3-NN+vae=====
mean F1-min: 0.737754019436 mean f-maj: 0.892114523254 mean accuracy: 0.847399626517
mean gmean: 0.773693986213 mean TPrate: 0.623717948718 mean AUC: 0.900381955002

0.001
=====NB+vae=====
mean F1-min: 0.723929788821 mean f-maj: 0.899492528504 mean accuracy: 0.85304388422
mean gmean: 0.758405872342 mean TPrate: 0.592948717949 mean AUC: 0.947490371947
=====NB+vae=====
mean F1-min: 0.722513056259 mean f-maj: 0.896839472444 mean accuracy: 0.849850606909
mean gmean: 0.755704556589 mean TPrate: 0.582692307692 mean AUC: 0.911666413297
=====NB+vae=====
mean F1-min: 0.673467983685 mean f-maj: 0.885160866564 mean accuracy: 0.830326797386
mean gmean: 0.71676239951 mean TPrate: 0.528846153846 mean AUC: 0.85504205939
=====C4.5+vae=====
mean F1-min: 0.791724386724 mean f-maj: 0.888046316702 mean accuracy: 0.854915966387
mean gmean: 0.831291008306 mean TPrate: 0.771794871795 mean AUC: 0.836984392419
=====C4.5+vae=====
mean F1-min: 0.80509101651 mean f-maj: 0.894321180115 mean accuracy: 0.864066293184
mean gmean: 0.842620214266 mean TPrate: 0.796153846154 mean AUC: 0.849559136516
=====C4.5+vae=====
mean F1-min: 0.812556551339 mean f-maj: 0.898878775022 mean accuracy: 0.869379084967
mean gmean: 0.851193018451 mean TPrate: 0.812820512821 mean AUC: 0.857497212932
=====3-NN+vae=====
mean F1-min: 0.739244702666 mean f-maj: 0.894662072007 mean accuracy: 0.850177404295
mean gmean: 0.774412065508 mean TPrate: 0.623717948718 mean AUC: 0.903269104084
=====3-NN+vae=====
mean F1-min: 0.739244702666 mean f-maj: 0.894662072007 mean accuracy: 0.850177404295
mean gmean: 0.774412065508 mean TPrate: 0.623717948718 mean AUC: 0.903269104084
=====3-NN+vae=====
mean F1-min: 0.739244702666 mean f-maj: 0.894662072007 mean accuracy: 0.850177404295
mean gmean: 0.774412065508 mean TPrate: 0.623717948718 mean AUC: 0.903269104084

0.0005
=====NB+vae=====
mean F1-min: 0.723929788821 mean f-maj: 0.899492528504 mean accuracy: 0.85304388422
mean gmean: 0.758405872342 mean TPrate: 0.592948717949 mean AUC: 0.948189672646
=====NB+vae=====
mean F1-min: 0.72138523671 mean f-maj: 0.897120741423 mean accuracy: 0.850014005602
mean gmean: 0.755214122353 mean TPrate: 0.583333333333 mean AUC: 0.913280379041
=====NB+vae=====
mean F1-min: 0.669120157598 mean f-maj: 0.882938644342 mean accuracy: 0.827385620915
mean gmean: 0.712505110285 mean TPrate: 0.520512820513 mean AUC: 0.855550065876
=====C4.5+vae=====
mean F1-min: 0.846935711631 mean f-maj: 0.91991656765 mean accuracy: 0.895028011204
mean gmean: 0.876638846396 mean TPrate: 0.828205128205 mean AUC: 0.880703354616
=====C4.5+vae=====
mean F1-min: 0.819384838994 mean f-maj: 0.904245645563 mean accuracy: 0.875093370682
mean gmean: 0.855180230404 mean TPrate: 0.803205128205 mean AUC: 0.859507702442
=====C4.5+vae=====
mean F1-min: 0.842916658221 mean f-maj: 0.914245860557 mean accuracy: 0.889472455649
mean gmean: 0.873761266412 mean TPrate: 0.834615384615 mean AUC: 0.87758437215
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
=====3-NN+vae=====
mean F1-min: 0.739244702666 mean f-maj: 0.894662072007 mean accuracy: 0.850177404295
mean gmean: 0.774412065508 mean TPrate: 0.623717948718 mean AUC: 0.903269104084
=====3-NN+vae=====
mean F1-min: 0.739244702666 mean f-maj: 0.894662072007 mean accuracy: 0.850177404295
mean gmean: 0.774412065508 mean TPrate: 0.623717948718 mean AUC: 0.903269104084

0.0002
=====NB+vae=====
mean F1-min: 0.706602008392 mean f-maj: 0.896284421602 mean accuracy: 0.847161531279
mean gmean: 0.745133461626 mean TPrate: 0.576282051282 mean AUC: 0.947097648728
=====NB+vae=====
mean F1-min: 0.729355161522 mean f-maj: 0.898508558076 mean accuracy: 0.852628384687
mean gmean: 0.761147873111 mean TPrate: 0.590384615385 mean AUC: 0.914165906557
=====NB+vae=====
mean F1-min: 0.673467983685 mean f-maj: 0.885160866564 mean accuracy: 0.830326797386
mean gmean: 0.71676239951 mean TPrate: 0.528846153846 mean AUC: 0.854169200365
=====C4.5+vae=====
mean F1-min: 0.825834296139 mean f-maj: 0.912842066692 mean accuracy: 0.88408496732
mean gmean: 0.857008814817 mean TPrate: 0.788461538462 mean AUC: 0.863400729705
=====C4.5+vae=====
mean F1-min: 0.805588150979 mean f-maj: 0.899078714995 mean accuracy: 0.867581699346
mean gmean: 0.842909819033 mean TPrate: 0.782051282051 mean AUC: 0.849227222053
=====C4.5+vae=====
mean F1-min: 0.811181398312 mean f-maj: 0.903337723742 mean accuracy: 0.872726423903
mean gmean: 0.847630902291 mean TPrate: 0.790384615385 mean AUC: 0.855271359076
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
=====3-NN+vae=====
mean F1-min: 0.734896876579 mean f-maj: 0.892439849784 mean accuracy: 0.847236227824
mean gmean: 0.770154776284 mean TPrate: 0.615384615385 mean AUC: 0.902890316206
time used: 3012.9421005290387
```

NB：比较占优的是0.005，比较占优的分类器是C4.5

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
=====NB+vae=====
mean F1-min: 0.716544167906 mean f-maj: 0.897553393564 mean accuracy: 0.849939309057
mean gmean: 0.752282115972 mean TPrate: 0.583974358974 mean AUC: 0.947126786257
=====NB+vae=====
mean F1-min: 0.718691545962 mean f-maj: 0.894415587554 mean accuracy: 0.846746031746
mean gmean: 0.751617187376 mean TPrate: 0.573717948718 mean AUC: 0.909412688761
=====NB+vae=====
mean F1-min: 0.673245544595 mean f-maj: 0.884783309111 mean accuracy: 0.830163398693
mean gmean: 0.715960169848 mean TPrate: 0.528205128205 mean AUC: 0.848213742779
time used: 607.2726537680792
```

##### 改变lamda1

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[2,1,0.001,0.001],
    'dataset_name':'ionosphere'
        }
1
=====NB+vae=====
mean F1-min: 0.730952380952 mean f-maj: 0.893288084465 mean accuracy: 0.847222222222
mean gmean: 0.759131963129 mean TPrate: 0.576923076923 mean AUC: 0.964882943144
=====NB+vae=====
mean F1-min: 0.7 mean f-maj: 0.884615384615 mean accuracy: 0.833333333333
mean gmean: 0.733799385705 mean TPrate: 0.538461538462 mean AUC: 0.887959866221
=====NB+vae=====
mean F1-min: 0.665789473684 mean f-maj: 0.876269956459 mean accuracy: 0.819444444444
mean gmean: 0.706582803096 mean TPrate: 0.5 mean AUC: 0.775919732441
2
=====NB+vae=====
mean F1-min: 0.730952380952 mean f-maj: 0.893288084465 mean accuracy: 0.847222222222
mean gmean: 0.759131963129 mean TPrate: 0.576923076923 mean AUC: 0.964882943144
=====NB+vae=====
mean F1-min: 0.7 mean f-maj: 0.884615384615 mean accuracy: 0.833333333333
mean gmean: 0.733799385705 mean TPrate: 0.538461538462 mean AUC: 0.886287625418
=====NB+vae=====
mean F1-min: 0.665789473684 mean f-maj: 0.876269956459 mean accuracy: 0.819444444444
mean gmean: 0.706582803096 mean TPrate: 0.5 mean AUC: 0.772575250836
3
=====NB+vae=====
mean F1-min: 0.696741854637 mean f-maj: 0.884942656308 mean accuracy: 0.833333333333
mean gmean: 0.73191538052 mean TPrate: 0.538461538462 mean AUC: 0.964882943144
=====NB+vae=====
mean F1-min: 0.665789473684 mean f-maj: 0.876269956459 mean accuracy: 0.819444444444
mean gmean: 0.706582803096 mean TPrate: 0.5 mean AUC: 0.886287625418
=====NB+vae=====
mean F1-min: 0.631578947368 mean f-maj: 0.867924528302 mean accuracy: 0.805555555556
mean gmean: 0.679366220487 mean TPrate: 0.461538461538 mean AUC: 0.76254180602
4
=====NB+vae=====
mean F1-min: 0.696741854637 mean f-maj: 0.884942656308 mean accuracy: 0.833333333333
mean gmean: 0.73191538052 mean TPrate: 0.538461538462 mean AUC: 0.963210702341
=====NB+vae=====
mean F1-min: 0.7 mean f-maj: 0.884615384615 mean accuracy: 0.833333333333
mean gmean: 0.733799385705 mean TPrate: 0.538461538462 mean AUC: 0.88127090301
=====NB+vae=====
mean F1-min: 0.631578947368 mean f-maj: 0.867924528302 mean accuracy: 0.805555555556
mean gmean: 0.679366220487 mean TPrate: 0.461538461538 mean AUC: 0.764214046823
5
=====NB+vae=====
mean F1-min: 0.696741854637 mean f-maj: 0.884942656308 mean accuracy: 0.833333333333
mean gmean: 0.73191538052 mean TPrate: 0.538461538462 mean AUC: 0.963210702341
=====NB+vae=====
mean F1-min: 0.7 mean f-maj: 0.884615384615 mean accuracy: 0.833333333333
mean gmean: 0.733799385705 mean TPrate: 0.538461538462 mean AUC: 0.884615384615
=====NB+vae=====
mean F1-min: 0.631578947368 mean f-maj: 0.867924528302 mean accuracy: 0.805555555556
mean gmean: 0.679366220487 mean TPrate: 0.461538461538 mean AUC: 0.755852842809
time used: 3099.99709921013
```

#### 不同数据集

```
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'breastw'
        }
breastw        
=====NB+vae=====
mean F1-min: 0.944333023881 mean f-maj: 0.968831247074 mean accuracy: 0.960058612545
mean gmean: 0.961515408072 mean TPrate: 0.967 mean AUC: 0.98575
=====NB+vae=====
mean F1-min: 0.949979128476 mean f-maj: 0.972248988752 mean accuracy: 0.964324206106
mean gmean: 0.964763824887 mean TPrate: 0.967 mean AUC: 0.985791666667
=====NB+vae=====
mean F1-min: 0.94776281642 mean f-maj: 0.971197095344 mean accuracy: 0.962895634678
mean gmean: 0.962635552165 mean TPrate: 0.962833333333 mean AUC: 0.987188405797
vehicle
=====NB+vae=====
mean F1-min: 0.547074564441 mean f-maj: 0.761575961252 mean accuracy: 0.688003104856
mean gmean: 0.722011847245 mean TPrate: 0.798947368421 mean AUC: 0.800193446356
=====NB+vae=====
mean F1-min: 0.476078952121 mean f-maj: 0.77100065445 mean accuracy: 0.682149438088
mean gmean: 0.654791674041 mean TPrate: 0.613421052632 mean AUC: 0.784271381579
=====NB+vae=====
mean F1-min: 0.433645416504 mean f-maj: 0.811188795412 mean accuracy: 0.717527926833
mean gmean: 0.603651487795 mean TPrate: 0.462631578947 mean AUC: 0.778064524291
segment-challenge
=====NB+vae=====
mean F1-min: 0.851616629799 mean f-maj: 0.974165054967 mean accuracy: 0.956033601493
mean gmean: 0.936913183215 mean TPrate: 0.912857142857 mean AUC: 0.980919584292
=====NB+vae=====
mean F1-min: 0.836738221475 mean f-maj: 0.976782247193 mean accuracy: 0.959371527623
mean gmean: 0.869876879728 mean TPrate: 0.766666666667 mean AUC: 0.980575006389
=====NB+vae=====
mean F1-min: 0.622493363504 mean f-maj: 0.958562439533 mean accuracy: 0.925374461087
mean gmean: 0.677659479669 mean TPrate: 0.469047619048 mean AUC: 0.980243206406
一次10折交叉验证的VAE：
=====NB+vae=====
mean F1-min: 0.736429750622 mean f-maj: 0.942737815573 mean accuracy: 0.905964709543
mean gmean: 0.926321276134 mean TPrate: 0.95619047619 mean AUC: 0.977074708237
=====NB+vae=====
mean F1-min: 0.757901745668 mean f-maj: 0.949137262336 mean accuracy: 0.915987377217
mean gmean: 0.932279036662 mean TPrate: 0.95619047619 mean AUC: 0.976706278218
=====NB+vae=====
mean F1-min: 0.765108315811 mean f-maj: 0.951733276809 mean accuracy: 0.919987555002
mean gmean: 0.930598838598 mean TPrate: 0.946428571429 mean AUC: 0.976465201465


diabetes
=====NB+vae=====
mean F1-min: 0.613975294048 mean f-maj: 0.816100841403 mean accuracy: 0.751332877649
mean gmean: 0.692681877805 mean TPrate: 0.567378917379 mean AUC: 0.793427350427
=====NB+vae=====
mean F1-min: 0.498347655367 mean f-maj: 0.80539023653 mean accuracy: 0.71995898838
mean gmean: 0.594702290773 mean TPrate: 0.402564102564 mean AUC: 0.753404558405
=====NB+vae=====
mean F1-min: 0.391366963574 mean f-maj: 0.797489968927 mean accuracy: 0.696633629528
mean gmean: 0.505359259159 mean TPrate: 0.283475783476 mean AUC: 0.721943019943
```

#### 数据集们

![1526871792669](/1526871792669.png)

##### 1breastw

cvae-gan

```
 {'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'breastw',
  'epochs': 1000,
  'hidden': [10, 10, 10, 10],
  'input_dim': 9,
  'lamda': [1, 1, 1, 1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.001},
=====NB+vae=====
mean F1-min: 0.947479355259 mean f-maj: 0.969577797466 mean accuracy: 0.961507304698
mean gmean: 0.966384608624 mean TPrate: 0.9835 mean AUC: 0.982633856683
=====NB+vae=====
mean F1-min: 0.945353504919 mean f-maj: 0.968502785522 mean accuracy: 0.960078733269
mean gmean: 0.964302121269 mean TPrate: 0.979333333333 mean AUC: 0.98340479066
=====NB+vae=====
mean F1-min: 0.945353504919 mean f-maj: 0.968502785522 mean accuracy: 0.960078733269
mean gmean: 0.964302121269 mean TPrate: 0.979333333333 mean AUC: 0.983450080515
```

cvae

```python
 {'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'breastw',
  'epochs': 1000,
  'hidden': [10, 10, 10, 10],
  'input_dim': 9,
  'lamda': [1, 1, 1, 1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.001}
=====NB+vae=====
mean F1-min: 0.940103871669 mean f-maj: 0.966675894665 mean accuracy: 0.957221590412
mean gmean: 0.957242235539 mean TPrate: 0.958666666667 mean AUC: 0.982943639291
=====NB+vae=====
mean F1-min: 0.940103871669 mean f-maj: 0.966675894665 mean accuracy: 0.957221590412
mean gmean: 0.957242235539 mean TPrate: 0.958666666667 mean AUC: 0.982762479871
=====NB+vae=====
mean F1-min: 0.941986224611 mean f-maj: 0.96782445896 mean accuracy: 0.958650161841
mean gmean: 0.958360008585 mean TPrate: 0.958666666667 mean AUC: 0.982632045089
```

##### 2vehicle

cvae-gan

```
'para': {'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'vehicle',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 18,
  'lamda': [2.5, 1, 0.1, 0.1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.1}
  =====NB+vae=====
mean F1-min: 0.55317122257 mean f-maj: 0.760851114071 mean accuracy: 0.689235597854
mean gmean: 0.726321192597 mean TPrate: 0.814210526316 mean AUC: 0.801189587551
=====NB+vae=====
mean F1-min: 0.545630939064 mean f-maj: 0.827051486122 mean accuracy: 0.750694542877
mean gmean: 0.705090628663 mean TPrate: 0.643684210526 mean AUC: 0.785676429656
=====NB+vae=====
mean F1-min: 0.504336505924 mean f-maj: 0.849617967525 mean accuracy: 0.770764233404
mean gmean: 0.646711343059 mean TPrate: 0.498157894737 mean AUC: 0.766760311235
```

cvae

```
{'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'vehicle',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 18,
  'lamda': [2.5, 1, 0.1, 0.1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.1}
=====NB+vae=====
mean F1-min: 0.52014347528 mean f-maj: 0.768635635018 mean accuracy: 0.697443218251
mean gmean: 0.689290108154 mean TPrate: 0.784473684211 mean AUC: 0.803445850202
=====NB+vae=====
mean F1-min: 0.513133672709 mean f-maj: 0.802604376014 mean accuracy: 0.72581924336
mean gmean: 0.680496910648 mean TPrate: 0.688947368421 mean AUC: 0.783883919534
=====NB+vae=====
mean F1-min: 0.486859245811 mean f-maj: 0.830169201336 mean accuracy: 0.750581823091
mean gmean: 0.643648398711 mean TPrate: 0.568421052632 mean AUC: 0.764519040992
```



##### 3segment-challenge

cvae-GAN

```python
{'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'segment-challenge',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'ini': 'he',
  'input_dim': 19,
  'lamda': [2, 1, 0.01, 0.01, 0.1],
  'latent_dim': 1,
  'learning_rate': 0.1},
=====NB+vae=====
mean F1-min: 0.81712479915 mean f-maj: 0.967840583955 mean accuracy: 0.945330903596
mean gmean: 0.922174033718 mean TPrate: 0.893333333333 mean AUC: 0.97822770253
=====NB+vae=====
mean F1-min: 0.781542240725 mean f-maj: 0.969574565754 mean accuracy: 0.946699853327
mean gmean: 0.83601523747 mean TPrate: 0.718571428571 mean AUC: 0.974167731493
=====NB+vae=====
mean F1-min: 0.573656969778 mean f-maj: 0.952519565613 mean accuracy: 0.91462731677
mean gmean: 0.648359594424 mean TPrate: 0.433095238095 mean AUC: 0.972433767783
```

cvae

```python
para_o = {
    'latent_dim':1,
    'epochs':1000,
    'batch_size':20, #为偶数
    'rate_1':0.1,#两个分类型网络的学习率
    'rate_2':0.1,#generator的学习率
    'PRE':sklearn.preprocessing.StandardScaler,#do not change it
#    'PRE':None,
    'hidden':[20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[2,1,0.01,0.01,0.1],
    'dataset_name':'segment-challenge'
        }
=====NB+vae=====
mean F1-min: 0.603591295689 mean f-maj: 0.939958521847 mean accuracy: 0.898706609183
mean gmean: 0.765307199636 mean TPrate: 0.775952380952 mean AUC: 0.955276854928
=====NB+vae=====
mean F1-min: 0.603527611374 mean f-maj: 0.947429218149 mean accuracy: 0.909444864216
mean gmean: 0.736031322444 mean TPrate: 0.708333333333 mean AUC: 0.95131953318
=====NB+vae=====
mean F1-min: 0.555451317821 mean f-maj: 0.948913583125 mean accuracy: 0.910098226588
mean gmean: 0.673574716909 mean TPrate: 0.610238095238 mean AUC: 0.949574495272
```

##### 4diabetes

cvae-gan

```
'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'diabetes',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 8,
  'lamda': [0.1, 1, 0.01, 0.01, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.0005},
  =====NB+vae=====
mean F1-min: 0.607878298472 mean f-maj: 0.789631071441 mean accuracy: 0.726572112098
mean gmean: 0.691950705141 mean TPrate: 0.607977207977 mean AUC: 0.793994301994
=====NB+vae=====
mean F1-min: 0.597554710475 mean f-maj: 0.781534986655 mean accuracy: 0.717447026658
mean gmean: 0.684284018851 mean TPrate: 0.607977207977 mean AUC: 0.784136752137
=====NB+vae=====
mean F1-min: 0.588712708168 mean f-maj: 0.774145153821 mean accuracy: 0.709620642515
mean gmean: 0.676253735059 mean TPrate: 0.603988603989 mean AUC: 0.768438746439
```

cvae

```
para_o =  {'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'diabetes',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 8,
  'lamda': [0.1, 1, 0.01, 0.01, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.0005}
=====NB+vae=====
mean F1-min: 0.277809485919 mean f-maj: 0.794870113515 mean accuracy: 0.680997949419
mean gmean: 0.407734841521 mean TPrate: 0.17905982906 mean AUC: 0.747943019943
=====NB+vae=====
mean F1-min: 0.237413751123 mean f-maj: 0.789525809132 mean accuracy: 0.670608339029
mean gmean: 0.372122994961 mean TPrate: 0.149287749288 mean AUC: 0.743763532764
=====NB+vae=====
mean F1-min: 0.234009222354 mean f-maj: 0.791931778733 mean accuracy: 0.673205741627
mean gmean: 0.366986590892 mean TPrate: 0.145584045584 mean AUC: 0.741165242165
```

##### 5ionosphere

cvae-gan

```
'para': {'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'ionosphere',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 34,
  'lamda': [2, 1, 1, 1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.0005},
=====NB+vae=====
mean F1-min: 0.682739321372 mean f-maj: 0.888490829157 mean accuracy: 0.83556022409
mean gmean: 0.724656067875 mean TPrate: 0.54358974359 mean AUC: 0.93867310226
=====NB+vae=====
mean F1-min: 0.671483910116 mean f-maj: 0.88488298602 mean accuracy: 0.830004668534
mean gmean: 0.715138917118 mean TPrate: 0.528205128205 mean AUC: 0.938661700618
=====NB+vae=====
mean F1-min: 0.656210800873 mean f-maj: 0.881439124916 mean accuracy: 0.824285714286
mean gmean: 0.702754461352 mean TPrate: 0.512179487179 mean AUC: 0.932826593696
```

cvae

```
{'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'ionosphere',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 34,
  'lamda': [2, 1, 1, 1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.0005}
=====NB+vae=====
mean F1-min: 0.558492116908 mean f-maj: 0.865833749846 mean accuracy: 0.795859010271
mean gmean: 0.607032240511 mean TPrate: 0.433333333333 mean AUC: 0.910532583359
=====NB+vae=====
mean F1-min: 0.543631435793 mean f-maj: 0.862312341283 mean accuracy: 0.78997665733
mean gmean: 0.595205701366 mean TPrate: 0.416666666667 mean AUC: 0.90980414513
=====NB+vae=====
mean F1-min: 0.531250483412 mean f-maj: 0.860883769854 mean accuracy: 0.787119514472
mean gmean: 0.583717484149 mean TPrate: 0.408974358974 mean AUC: 0.908710854363
```

##### 6sonar

cvae-gan

```
'PRE': sklearn.preprocessing.data.StandardScaler,
  'batch_size': 20,
  'dataset_name': 'sonar',
  'epochs': 1000,
  'hidden': [20, 10, 20, 10],
  'input_dim': 60,
  'lamda': [2.5, 1, 1, 1, 0.1],
  'latent_dim': 2,
  'learning_rate': 0.01},
=====NB+vae=====
mean F1-min: 0.592465604807 mean f-maj: 0.637107508008 mean accuracy: 0.632922077922
mean gmean: 0.593528641035 mean TPrate: 0.583333333333 mean AUC: 0.708611111111
=====NB+vae=====
mean F1-min: 0.435035014006 mean f-maj: 0.712141271245 mean accuracy: 0.622705627706
mean gmean: 0.505917821749 mean TPrate: 0.323333333333 mean AUC: 0.723838383838
=====NB+vae=====
mean F1-min: 0.249427239427 mean f-maj: 0.711636871656 mean accuracy: 0.58645021645
mean gmean: 0.340263708188 mean TPrate: 0.162222222222 mean AUC: 0.716186868687
```

