

# pytorch中多线程的共享优化器设置
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import math


# from collections import defaultdict
# state=defaultdict(torch.FloatTensor)#defaultdict用法,因为字典里没有state这个key,会报错,但是用defaultdict就可以报类型
# print(state['state'])

class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]  # 根据p的类型来创建的字典
                state['step'] = torch.zeros(1)
                state['exp_avg'] =p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data  # 梯度参数
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1-beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                #p.data.addcdiv_(-step_size, exp_avg, denom)
                h = p.data
                step_size = float(step_size)
                torch.Tensor.addcdiv_(h, -step_size, exp_avg, denom)
                p.data = h

        return loss




