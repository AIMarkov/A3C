"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):#numpy to tensorFloat
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):#initial weight
    for layer in layers:
        nn.init.normal(layer.weight, mean=0., std=0.1)
        nn.init.constant(layer.bias, 0.1)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):#optimizer,local net,global net,done,netx state,0.9


    if done:
        v_s_ = 0.               # terminal,而且这个地方用的是Ｖ值不是Ｑ值
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0] #s_[None, :],(3,)->(1,3)#取最后一个状态,求其v值(因为5-step,而且整个过程都由local来产生的,因此用local_net来计算)


    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_#use n step #这个地方计算的就是目标v值
        buffer_v_target.append(v_s_)

    buffer_v_target.reverse()
    #have problem
    # print('bs:',bs)
    # print('np.vstack:',np.vstack(bs))#(5,3)
    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),  #np.vstack() change the dimension,vertical stack
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        #gp._grad=lp.grad
        gp.grad = lp.grad#取得是grade
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())#这里是将全球网的参数赋值给


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )