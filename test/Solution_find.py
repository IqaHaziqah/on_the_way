# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        s = list(s)
        count=len(s)
        for i in range(0,count):
            if s[i]==' ':
                s[i]='%20'
        return ''.join(s)
    
#while True:
#    try:
#        s = list(eval(input()))
#        t = Solution()
#        ss = (t.replaceSpace(s))
#    except:
#        break
        
    
s = "hello world"
t = Solution()
ss = t.replaceSpace(s)
print(ss)



class Solution:
    # matrix类型为二维列表，需要返回列表
    #未能考虑非方阵情况
    def printMatrix(self, matrix):
        high = len(matrix)**0.5-1
        n = int(len(matrix)**0.5)-1
        low = 0
        i = 0
        j = 0
        value = []
        while high>=low:
            print(matrix[i*n+j])
            while i<j:
                if j<high:
                    j+=1
                else:
                    i+=1
                print(matrix[i*n+j])
                print('hello world')
            low += 1
            while i>j:
                if j>low:
                    j-=1
                else:
                    i-=1
                print(matrix[i*n+j])
                print('hellow')
            high-=1
            
        return
c = Solution()
c.printMatrix([[1]])