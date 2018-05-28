剑指offer

把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

$M = 2^i*3^j*5^m$

并不断判断应该在哪个位置上加1。

```
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index==0:
            return 0
        ans = [1]
        num_2 = 0
        num_3 = 0
        num_5 = 0
        for i in range(index):
            ans.append(min(min(ans[num_2]*2,ans[num_3]*3),ans[num_5]*5))
            if ans[-1]%2==0:
                num_2+=1
            if ans[-1]%3==0:
                num_3+=1
            if ans[-1]%5==0:
                num_5+=1
        return ans[index-1]
```

```
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        Node = ListNode(0)
        pre = Node
        point = pHead
        next1 = point.next
        while next1:
            
            value = point.val
            if next1.val != value and point.val!=next1.val:
                print(value)
                print(pre.val,point.val,next1.val)
                pre.next = point
                pre = point 
                point = next1
                next1 = point.next
            else:
                point = next1
                next1 = point.next
        return Node.next
c = Solution()
point = ListNode(0)
point1 = point
for vlaue in {1,2,3,3,4,4,5}:
    d = ListNode(vlaue)
    point.next = d
    point = d
c.deleteDuplication(point1.next)
```

```python
连续子向量的最大和
在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return 0
        maxs = array[0]
        curs = 0
        for i in range(len(array)):
            curs += array[i]
            if curs> maxs:
                maxs = curs
            if curs <=0:
                curs = 0
        return maxs
```

```
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。

```

![img](F:\OneDrive\笔记\image\961875_1469289666488_886555C4C4726220976FEF4D3A32FFCD)

![1527240715786](C:\Users\zhouying\AppData\Local\Temp\1527240715786.png)



理论知识

> ```
> 在多线程环境下，每个线程拥有一个栈和一个程序计数器。栈和程序计数器用来保存线程的执行历史和线程的执行状态，是线程私有的资源。其他的资源（比如堆、地址空间、全局变量）是由同一个进程内的多个线程共享
> ```

> ```
> 进程在退出时会自动关闭自己打开的所有文件
> 关闭自己打开的网络链接
> 销毁自己创建的所有线程
> ```

