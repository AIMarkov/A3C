import torch
import numpy as np
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn as nn
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()
        self.Rnn=nn.RNN(input_size=20,hidden_size=50,num_layers=2,batch_first=True)
        self.linear1=nn.Linear(50,20)
    def forward(self,input):
        h0=Variable(torch.zeros(2,4,50))
        return self.Rnn(input,h0)

class process(mp.Process):
    def __init__(self):
        super(process,self).__init__()
        self.NN=Rnn()
        self.input=input
    def run(self,input=Variable(torch.randn(4,24,20))):
        out1,out2=self.NN(input)
        print(out1)
        print("must define as run")


x=Variable(torch.randn(4,24,20))

worker1=process()
worker1.start()#can't give the arguments
worker1.join()
x=torch.FloatTensor(x.data)
print('x:',x)
y=Variable(torch.zeros(2,3))
c=Variable(torch.ones(2,3))

b=c-y
print(b.pow(2))
d=torch.ones(2,3)
distribution = torch.distributions.Normal
m = distribution(0.1, 0.1)
#h=torch.log(m.scale)
c=m.log_prob(d)
print(c)
#print(h)
m=torch.distributions.Normal(0.1,0.1).scale