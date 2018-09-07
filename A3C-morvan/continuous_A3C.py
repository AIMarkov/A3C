"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"#get the variable of system,分线程执行数

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000#最大情节数
MAX_EP_STEP = 200#情节最大步数

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
print('N_A:',N_A)#1
print('N_S:',N_S)#3

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 100)
        self.mu = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x=Variable(x)
        out=self.a1(x)
        a1 = F.relu(out)
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):

        self.train()
        mu, sigma, values = self.forward(s)
        # print('values:',values)
        # print('v_t:',v_t)
        #have problem
        v_t=Variable(v_t)#change type
        #values=torch.FloatTensor(values.data)
        td = v_t - values

        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        #print('m:',m)
        #print('a:',a)
        a=Variable(a)
        log_prob = m.log_prob(a)

        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(sigma)#torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        #print('every thing is good')
        a_loss = -exp_v

        total_loss = (a_loss + c_loss).mean()

        return total_loss


class Worker(mp.Process):#define a inherit class
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt#global net and global optimizer
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:#所有进程总情节数小于3000
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []#buffer
            ep_r = 0. #return
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :])) #type change for s

                s_, r, done, _ = self.env.step(a.clip(-2, 2)) #excute action
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)#1,(1,),numpy,float
                buffer_s.append(s)#3,(3,),numpy
                buffer_r.append((r+8.1)/8.1)    # number r,normalize,here because know averag reward


                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net,every five step as a batch for updating
                    # sync

                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)#update

                    buffer_s, buffer_a, buffer_r = [], [], []


                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()#number of ep.return

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]#根据核数创建进程数

    [w.start() for w in workers] #run
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
