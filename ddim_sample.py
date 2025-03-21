#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/20 10:55
# @Author  : jianqiugao
# @File    : ddim_sample.py
# @email: 985040320@qq.com
import torch
import matplotlib.pyplot as plt
from diffusion_model import Net

if __name__ == '__main__':
    device = torch.device('cuda:0')
    time_step = 1000
    beta = torch.linspace(0.0001, 0.002, time_step).to(device)
    alpha_t = 1 - beta
    alpha_t_hat = torch.cumprod(alpha_t, dim=-1).to(device)
    alphas_cumprod_prev = torch.roll(alpha_t_hat,1)
    alphas_cumprod_prev[0] = 1
    eta = 0
    segama = torch.sqrt(eta*torch.sqrt((1-alphas_cumprod_prev)/(1-alpha_t_hat))*torch.sqrt(1-alpha_t_hat/alphas_cumprod_prev))
    one_minus_alpha_t = torch.sqrt(1-alphas_cumprod_prev)

    x_len = 128
    time_dim = x_len
    c = torch.tensor([[0.5, 0.0113]]).to(device)
    model = Net([x_len,256,512,1024,2048,4096], time_dim, steps=time_step).to(device)
    model.load_state_dict(torch.load('model.pt'))

    fig, axes = plt.subplots(3, 1)

    xt = torch.randn(1, time_dim).to(device)
    print(xt.mean().item(), xt.var().item())
    axes[0].plot(xt.detach().cpu().reshape(-1))
    axes[2].plot(xt.detach().cpu().reshape(-1))

    for i in reversed(list(range(time_step))[::10]):
        if i > 1:
            zt = torch.randn(1, time_dim).to(device)
        else:
            zt = torch.zeros(1, time_dim).to(device)

        eps = model(xt, torch.tensor([i]).to(device), c)
        xt = torch.sqrt(alphas_cumprod_prev[i]) * (xt - eps * one_minus_alpha_t[i])/torch.sqrt(alpha_t_hat[i]) + segama[i]*zt # + torch.sqrt(1-alphas_cumprod_prev[i]- segama[i]**2) * eps # +  segama[i]*zt #  # 这里方差部分的影响很小

    print(xt.mean().item(), xt.var().item())
    axes[1].plot(xt.detach().cpu().reshape(-1))
    axes[2].plot(xt.detach().cpu().reshape(-1))
    plt.show()

