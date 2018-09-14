import math, os
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
os.environ["OMP_NUM_THREADS"]='1'
c=os.environ["OMP_NUM_THREADS"]
# print(c)
# print(os.environ.keys())
# print(mp.cpu_count())
# print(mp.Value('i', 0))
# print(mp.Value('d', 0.))
# print(mp.Queue())
# a=Variable(torch.FloatTensor([[2,3,4,5]]))
# b=Variable(torch.FloatTensor([[0.1,0.1,0.1,0.1]]))
# distribution=torch.distributions.Normal(a.data,b.data)
# print(a)
# print(distribution.sample())
# a=np.asarray([[2.4,3,3.6,4.5,5]])
# print(np.clip(a,2,3))
# a=np.asarray([[2.4,3,3.6,4.5,5]])
# from collections import defaultdict
# state=defaultdict(torch.FloatTensor)
# print(state['state'])

#
# d=torch.zeros(100,3)
# print(d)
# c=torch.randn(100,3)
# d=torch.randn(100,3)
# # torch.Tensor.addcdiv_(d,-1.126762,c,d)
# #print(c.add_(-1,d))
# print(c)
# print(d)
# print(d.mul_(0.9))
# print(d.add_(1 - 0.9, c))
# c=[1,2,3]
# print(del(c[0]))
#multiprocessing shared list queue
#解决共享buffer问题
import random
import time
import torch.multiprocessing as mp
import queue





def sample0(q):
    with open('1.txt','a+') as f:
        for i in range(100):
            with q.get_lock():
                q.value+=1
                f.writelines('0sample'+str(q.value)+'\n')



def sample1(q):
    with open('1.txt','a+') as f:
        for i in range(100):
            with q.get_lock():
                q.value+=1
                f.writelines('1sample'+str(q.value)+'\n')



q=mp.Value('i',0)

t1=mp.Process(target=sample0,args=(q,))
t2=mp.Process(target=sample1,args=(q,))
t1.start()
t2.start()

t1.join()
t2.join()

