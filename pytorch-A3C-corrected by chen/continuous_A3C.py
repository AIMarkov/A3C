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
MAX_EP = 100000#最大情节数
MAX_EP_STEP = 200#情节最大步数

#env=gym.make('Pendulum-v0')
#env = gym.make('HumanoidStandup-v2')
env = gym.make('HalfCheetah-v2')
#env = gym.make('Walker2d-v2')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
action_high=float(env.action_space.high[0])
action_low=float(env.action_space.low[0])
print('N_A:',N_A)#1
print('N_S:',N_S)#3
print('scale:',action_high,action_low)


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim #状态维度
        self.a_dim = a_dim #动作维度
        self.a1 = nn.Linear(s_dim, 100)#
        self.mu = nn.Linear(100, a_dim)#动作维度输出均值
        self.sigma = nn.Linear(100, a_dim)#动作维度输出方差
        self.c1 = nn.Linear(s_dim, 100)#中途值的输出
        self.v = nn.Linear(100, 1)#v值
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])#初始化权重
        self.distribution = torch.distributions.Normal#建立一个正太分布对象

    def forward(self, x):
        x=Variable(x) #状态

        a1 = F.relu(self.a1(x))#

        mu = 2 * F.tanh(self.mu(a1))

        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0

        c1 = F.relu(self.c1(x))

        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.data, sigma.data)#根据正太分布来选择动作,tensor.View([2,-1]) Returns a new tensor with the same data as the self tensor but of a different size.
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):

        self.train()#表示现在是训练模式
        mu, sigma, values = self.forward(s)

        v_t=Variable(v_t)#change type

        td = v_t - values#批量是5但是，每次这种训练是连续的，并没有打破相关性，每5步训练一次，因此相关性并没有打破

        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)

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
        #self.env = gym.make('Pendulum-v0').unwrapped
        #self.env = gym.make('HumanoidStandup-v2').unwrapped
        self.env = gym.make('HalfCheetah-v2').unwrapped
        #self.env = gym.make('Walker2d-v2').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:#所有进程总情节数小于3000
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []#buffer
            ep_r = 0. #return
            for t in range(MAX_EP_STEP):
                # if self.name == 'w0':
                #     self.env.render()

                a = self.lnet.choose_action(v_wrap(s[None, :])) #type change for s

                s_, r, done, _ = self.env.step(a.clip(action_low, action_high)) #将excute action
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)#1,(1,),numpy,float
                buffer_s.append(s)#3,(3,),numpy
                #buffer_r.append((r+8.1)/8.1)    # number r,normalize,here because know averag reward
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net,every five step as a batch for updating
                    # sync
                    # print('buffer_a:',len(buffer_a))
                    # print('buffer_s:',len(buffer_s))
                    # print('buffer_r:',len(buffer_r))

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
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer,事实上优化器可以不用定义全局的，每个线程一个就可以，避免后面出错

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()#number of ep.return

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]#mp.cpu_count()根据核数创建进程数

    [w.start() for w in workers] #run这几个从这儿开始同时运行
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
