#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 9:34
# @Author  : jianqiugao
# @File    : diffusion_model.py
# @email: 985040320@qq.com
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