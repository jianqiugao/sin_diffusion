#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 9:34
# @Author  : jianqiugao
# @File    : diffusion_model.py
# @email: 985040320@qq.com
import math
import torch
from torch import nn


class resblock(nn.Module):
    def __init__(self,dims:list,normal=True):
        super().__init__()
        self.shortcut = nn.Linear(dims[0],dims[1])
        self.dims = dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(dims[0],dims[0]),
                                     nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(dims[0],dims[1]),
                                     nn.ReLU(),))
        self.layers.append(nn.Sequential(nn.Linear(dims[1],dims[1])))
        self.normals = nn.ModuleList()
        if normal:
            self.normals.append(nn.Identity())
            self.normals.append(nn.Identity())
            self.normals.append(nn.Identity())
            # self.normals.append(nn.LayerNorm(dims[0]))
            # self.normals.append(nn.LayerNorm(dims[1]))
            # self.normals.append(nn.LayerNorm(dims[1]))
            self.total_normal = nn.LayerNorm(dims[1])

        else:
            self.normals.append(nn.Identity())
            self.normals.append(nn.Identity())
            self.normals.append(nn.Identity())
            self.total_normal = nn.Identity()

    def forward(self,x):
        shortcut = x
        for module,normal in zip(self.layers, self.normals):
            x = module(x)
            x = normal(x)
        x = self.total_normal(x + self.shortcut(shortcut))
        return x


class Net(nn.Module):
    def __init__(self,embed_dim_list,time_dim,steps):
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.time_dim = time_dim
        self.embed_t = torch.nn.Embedding(steps, self.time_dim)
        self.embed_t = torch.nn.Sequential(nn.Linear(1,time_dim),
                                           nn.ReLU(),
                                           nn.Linear(time_dim,time_dim))
        self.embed_c = torch.nn.Sequential(nn.Linear(2,time_dim),
                                           nn.ReLU(),
                                           nn.Linear(time_dim,time_dim))
        self.revsed_embed_dim_list = list(reversed(embed_dim_list))

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for index in range(len(self.embed_dim_list)-1):
            self.encoder.append(self._build_mlp([self.embed_dim_list[index],self.embed_dim_list[index+1]],normal=True))

        for index in range(len(self.embed_dim_list)-1):
            if index<len(self.embed_dim_list)-2:
                self.decoder.append(self._build_mlp([self.revsed_embed_dim_list[index]*2+ self.time_dim*2 ,self.revsed_embed_dim_list[index+1]],normal=True))
            else:
                self.decoder.append(self._build_mlp(
                    [self.revsed_embed_dim_list[index] * 2 + self.time_dim*2, self.revsed_embed_dim_list[index + 1]]))

    def _build_mlp(self, dim:list,normal=False):
        if normal:

            return resblock(dim,normal=True)
        else:
            print('run',dim[0],dim[1])

            return resblock(dim,normal=False)

    def forward(self, x, t,c):
        t = (t.reshape(-1,1)+1.)/1000.
        embed_t = self.embed_t(t)
        embed_c = self.embed_c(c)
        contex = []
        for moudule in self.encoder:
            x = moudule(x)
            contex.append(x)
        for moudule in self.decoder:
            x = torch.concat([x, embed_t, embed_c, contex.pop(-1)],dim=-1)
            x = moudule(x)
        return x


class att_trans(nn.Module):
    def __init__(self,embed_dim,max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.input_dim = embed_dim*3
        self.q = nn.Linear(self.input_dim,embed_dim)
        self.k = nn.Linear(self.input_dim,embed_dim)
        self.v = nn.Linear(self.input_dim,embed_dim)
        self.o = nn.Sequential(nn.Linear(embed_dim,embed_dim),
                               nn.ReLU(),
                               nn.Linear(embed_dim, embed_dim)
                               )
        # self.t = nn.Linear(embed_dim,embed_dim)
        # self.c = nn.Linear(embed_dim,embed_dim)
        self.normal = nn.LayerNorm(self.embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # 填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 增加一个批次维度
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    def forward(self,x,t,c):
        seq_len = x.shape[1]
        # t = self.t(t).repeat_interleave(seq_len,1)
        # c = self.c(c).repeat_interleave(seq_len,1)
        t = t.repeat_interleave(seq_len, 1)
        c = c.repeat_interleave(seq_len, 1)
        pe = self.pe[:seq_len,0,:]
        x = x + pe.unsqueeze(0)
        cond = torch.concat([x,t,c],dim=-1)
        q, k, v = self.q(cond),self.q(cond),self.q(cond)
        att_socre = q@k.permute(0,2,1)/self.embed_dim
        v = att_socre@v
        out = self.o(v)
        out = self.normal(out)+x
        return out



class att_net(nn.Module):
    """
    改为注意力来做，不能一致用mlp
    """
    pass
    def __init__(self,embed_dim_list,time_dim,steps):
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.time_dim = time_dim
        # self.embed_t = torch.nn.Embedding(steps, self.time_dim)
        self.embed_t = torch.nn.Sequential(nn.Linear(1,time_dim),
                                           nn.LeakyReLU(),
                                           nn.Linear(time_dim,time_dim))
        self.embed_c = torch.nn.Sequential(nn.Linear(2,time_dim),
                                           nn.LeakyReLU(),
                                           nn.Linear(time_dim,time_dim))
        self.embed_x = torch.nn.Sequential(nn.Linear(1,time_dim),
                                           nn.LeakyReLU(),
                                           nn.Linear(time_dim,time_dim))
        self.output = torch.nn.Sequential(nn.Linear(time_dim,time_dim),
                                          nn.LeakyReLU(),
                                          nn.Linear(time_dim,1,bias=False))
        self.encoder_decoder = nn.Sequential()
        for i in range(len(self.embed_dim_list)):
            self.encoder_decoder.append(att_trans(embed_dim_list[0]))

    def forward(self,x, t, c):
        t = (t.reshape(-1, 1) + 1.) / 1000. # 是一个时间
        c = c.reshape(-1,2)
        x = self.embed_x(x.unsqueeze(-1))
        embed_t = self.embed_t(t).unsqueeze(1)
        embed_c = self.embed_c(c).unsqueeze(1)
        for item in self.encoder_decoder:
            x = item(x,embed_t,embed_c)
        x = self.output(x).squeeze(-1)
        return x