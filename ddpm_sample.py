#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 9:34
# @Author  : jianqiugao
# @File    : ddpm_sample.py
# @email: 985040320@qq.com
import torch
import matplotlib.pyplot as plt
from diffusion_model import Net

if __name__ == '__main__':
    device = torch.device('cuda:0')
    # torch.manual_seed(0)
    time_step = 1000
    beta = torch.linspace(0.0001, 0.002, time_step).to(device)
    alpha_t = 1 - beta
    alpha_t_hat = torch.cumprod(alpha_t, dim=-1).to(device)
    alpha_t_hat_s = torch.sqrt(alpha_t_hat).to(device=device)
    one_minus_alpha_t_sqrt = torch.sqrt(1 - alpha_t_hat).to(device=device)
    alphas_cumprod_prev = torch.roll(alpha_t_hat,1)
    alphas_cumprod_prev[0] = 1
    var = (beta * (1.0 - alphas_cumprod_prev) / (1.0 - alpha_t_hat))

    x_len = 128
    time_dim = x_len
    c = torch.tensor([[0.0193,  0.0113]]).to(device)
    model = Net([x_len,256,512,1024,2048,4096], time_dim, steps=time_step).to(device)
    model.load_state_dict(torch.load('model.pt'))

    alphas_cum_prev = torch.cat((torch.tensor([1.0]).to(device), alpha_t_hat[:-1]), 0).to(device)
    posterior_variance = (beta * (1 - alphas_cum_prev) / (1 - alpha_t_hat)).to(device)

    fig, axes = plt.subplots(3, 1)

    xt = torch.randn(1, time_dim).to(device)
    print(xt.mean().item(),xt.var().item())
    axes[0].plot(xt.detach().cpu().reshape(-1))
    axes[2].plot(xt.detach().cpu().reshape(-1))

    for i in reversed(range(time_step)):
        if i > 1:
            zt = torch.randn(1, time_dim).to(device)
        else:
            zt = torch.zeros(1, time_dim).to(device)

        coeff_1 =(1. / torch.sqrt(alpha_t[i]))

        eps = model(xt, torch.tensor([i]).to(device), c)
        eps_ = (1 - alpha_t[i]) / (one_minus_alpha_t_sqrt[i])
        xt = coeff_1 * (xt - eps*eps_) #+ torch.sqrt(beta[i]) * zt # 这里方差部分的影响很小

    print(xt.mean().item(),xt.var().item())
    axes[1].plot(xt.detach().cpu().reshape(-1))
    axes[2].plot(xt.detach().cpu().reshape(-1))
    plt.show()