import torch
import numpy as np
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn as nn
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()
        self.Rnn=nn.RNN(input_size=20,hidden_size=50,num_layers=2,batch_first=True)
    def forward(self,input,q,A):
        h0=Variable(torch.zeros(2,4,50))
        out1,out2=self.Rnn(input,h0)
        q.put(out1)
        q.put(out2)
        for i in range(10):
            A.value+=i

Rnn1=Rnn()
x=Variable(torch.randn(4,24,20))
q=mp.Queue()
A=mp.Value('i',0)
mp1=mp.Process(target=Rnn1,args=(x,q,A))
mp1.start()
mp1.join()
print(A.value)
print(q.get())
print(q.get())


