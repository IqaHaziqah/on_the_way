编程经验

> 1. 要能看懂函数和类的输入和输出，尤其是函数的输入说明等，看懂函数的抛出异常并明白异常抛出原因，从而尽快对bug进行排除。
> 2. **object is not callable:在python中一般是将变量作为函数使用，会导致该类问题出现，debug时检查变量类型即可。
> 3. ​

#### python系统函数

统计字符串中不同的字符出现的个数

```python
import string  
s = raw_input('input a string:\n')  
letters = 0  
space = 0  
digit = 0  
others = 0  
for c in s:  
    if c.isalpha():  
        letters += 1  
    elif c.isspace():  
        space += 1  
    elif c.isdigit():  
        digit += 1  
    else:  
        others += 1  
print 'char = %d,space = %d,digit = %d,others = %d' % (letters,space,digit,others)  
```



#### python 使用copy

在copy模块中有两种copy方式，copy和deepcopy，在单一变量中，两种method并没有区别，但在复杂变量中，例如list中嵌套等，简单的copy只是在第一次复制的时候copy了内部嵌套变量的地址，导致复制后的变量同原始变量间并未独立，因此复制后产生的变量会随着原始变量的变化而变化，在此种情况下，需要使用deepcopy来产生独立的复制变量。

#### python 使用numpy

生成数列：

```python
import numpy as np
#arange(start, stop=None, step=1, dtype=None)
#产生一个在区间 [start, stop) 之间，以 step 为间隔的数组，如果只输入一个参数，则默认从 0 开始，并以这个值为结束
i = np.arange(start,stop,step)
#linspace(start,stop,N)产生[start,stop]间等间隔的N个数
#logspace(start,stop,N)产生[start,stop]间10的等间隔的幂次的数
```

#### python 使用threading

主要使用threading的以下几个函数

threading.current_thread() #返回当前线程id

threading.Thread(target,*args) #开启一个新线程，target为该线程需要执行的函数，args为对应函数需要的参数

threading.active_count() #当前激活的线程id

```python
a = threading.Thread(target=function)
a.start()#开始执行线程所指向的代码
a.join()#等待a执行完毕后，再执行以下代码，否则在a.start()语句后，会马上执行之后的代码
```

#### python 使用logging

| Level      | When it’s used                                               |
| ---------- | ------------------------------------------------------------ |
| `DEBUG`    | Detailed information, typically of interest only when diagnosing problems. |
| `INFO`     | Confirmation that things are working as expected.            |
| `WARNING`  | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. |
| `ERROR`    | Due to a more serious problem, the software has not been able to perform some function. |
| `CRITICAL` | A serious error, indicating that the program itself may be unable to continue running. |

| Level      | Numeric value |
| ---------- | ------------- |
| `CRITICAL` | 50            |
| `ERROR`    | 40            |
| `WARNING`  | 30            |
| `INFO`     | 20            |
| `DEBUG`    | 10            |
| `NOTSET`   | 0             |

logging.basicConfig(level=logging.INFO) 该语句用于定义logging的级别，当出现该语句时，意味着该级别和以上级别的logging会被跟踪。

```python
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
DEBUG:root:This message should go to the log file
INFO:root:So should this
WARNING:root:And this, too
```

使用logging追踪变量结果：一般使用%s，%f，%d来跟踪变量结果。

```python
import logging
logging.warning('%s before you %s', 'Look', 'leap!')
```

使用logging和文件来检测程序中的变量

```python
import logging
logging.basicConfig(level=logging.INFO,filename='log',filemode='w')
para = {}
for value1,value2 in zip([1,2,3,4,5],[1,2,3,4,5]):
    para['value1']=value1
    para['value2']=value2
    print(para)
    logging.warning('the para is %s',para)
the result in log:
WARNING:root:the para is {'value2': 1, 'value1': 1}
WARNING:root:the para is {'value2': 2, 'value1': 2}
WARNING:root:the para is {'value2': 3, 'value1': 3}
WARNING:root:the para is {'value2': 4, 'value1': 4}
WARNING:root:the para is {'value2': 5, 'value1': 5}
```



#### python 使用tensorflow 

直接区分self.label中出现的值，然后分批喂入数据，计算其对应的feature center，考虑使用tf.where

##### loss函数的定义及使用

> in papers people write min(log1-D), but in practice folks practically use max logD,最终的generator的loss函数为real=fake，fake=real
>
> Use SGD for discriminator and ADAM for generator 

https://sinpycn.github.io/2017/05/09/GAN-Tutorial-Tips-and-Tricks.html

单侧标签平滑：real=0.9而不是1，这种做法可以避免判别器的极端预测行为： 如果判别器通过学习来预测一个极端大的逻辑值，也就是对某些输入的输出概率接近于1时， 它将被惩罚并被鼓励回到一个较小的逻辑值上去。

判别器应该总是最优的？

https://zhuanlan.zhihu.com/p/28487633

##### tensorboard

##### gradient check

https://blog.csdn.net/han_xiaoyang/article/details/50521064 参考该博客中的理论

http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96  梯度检验与高级优化

使用双精度gradient check后实现正常

##### 模型的保存与恢复

故事：在训练CVAE-GAN的时候，希望在vae部分采用原本训练好的vae初始化CVAE-GAN中的参数，因而出现这部分的操作。

直接在初始化后，利用vae架构先训练直到网络稳定，然后再对CVAE-GAN进行训练