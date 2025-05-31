import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import os
import time
from PIL import Image
import pickle
import sys
class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

class Sine_kx(nn.Module):
    def __init__(self,k=30.0):
        super().__init__()
        self.k = k

    def forward(self, input):
        return torch.sin(self.k  * input)

class Tanh_kx(nn.Module):
    def __init__(self, k=1.0):
        super(Tanh_kx, self).__init__()
        self.k = k  # 将 k 设置为模块的属性，而不是可学习的参数

    def forward(self, x):
        return torch.tanh(self.k * x)


class deepsdf_like_network(nn.Module):
    """
    与deepsdf类似，在某些层会将输入uv与上一层concate后作为本层输入
    """
    def __init__(self,item_embedding_dim:int=128,uv_dim:int=32,out_dim:int=3,
                 mlp_out_dim_list=[256,256,128,128,64,32],
                 nonlinearity='tanh_kx', tanh_k=3,dropout_rate=0.3,
                 use_weight_norm:bool=True):
        super().__init__()
        self.layers = nn.ModuleList()

        activation_functions = {
            'relu': nn.ReLU(inplace=True),
            'sin': Sine(),
            "leaky_relu":nn.LeakyReLU(negative_slope=0.01),
            "tanh":nn.Tanh(),
            "sigmoid":nn.Sigmoid(),
            "tanh_kx":Tanh_kx(k=tanh_k),
            "sine_kx":Sine_kx(k=tanh_k),            
        }

        if nonlinearity not in activation_functions:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        activation_function   = activation_functions[nonlinearity]
        self.num_of_layers    = len(mlp_out_dim_list)+1
        self.mlp_out_dim_list = mlp_out_dim_list
        if use_weight_norm==True:
            self.layers.append(weight_norm(nn.Linear((item_embedding_dim+uv_dim), mlp_out_dim_list[0]),name="weight"))
        else:
            self.layers.append(nn.Linear((item_embedding_dim+uv_dim), mlp_out_dim_list[0]))
        self.layers.append(activation_function)
        self.layers.append(nn.Dropout(dropout_rate))        
        for i in range(0,self.num_of_layers-3):
            if use_weight_norm==True:
                self.layers.append(weight_norm(nn.Linear(mlp_out_dim_list[i]+uv_dim, mlp_out_dim_list[i+1]),name="weight"))
            else:
                self.layers.append(nn.Linear(mlp_out_dim_list[i]+uv_dim, mlp_out_dim_list[i+1]))
            self.layers.append(activation_function)
            self.layers.append(nn.Dropout(dropout_rate))  
        if use_weight_norm==True:
            self.layers.append(weight_norm(nn.Linear(mlp_out_dim_list[-2]+uv_dim, mlp_out_dim_list[-1]),name="weight"))
        else:
            self.layers.append(nn.Linear(mlp_out_dim_list[-2]+uv_dim, mlp_out_dim_list[-1]))
        self.layers.append(activation_function)
        self.layers.append(nn.Dropout(dropout_rate))  

        self.layers.append(nn.Linear(mlp_out_dim_list[-1],out_dim))
        # self.layers.append(activation_function)
        # self.layers.append(nn.Dropout(dropout_rate))                     
            
    def forward(self,item_embedding:torch.tensor,
                uv:torch.tensor):
        """
        date          : 2023.09.11
        input         : 
        item_embedding: tensor [bs*N,item_embedding_dim]/[bs,N,item_embedding_dim]
        uv            : tensor [bs*N,uv_dim]/[bs,N,uv_dim]
        output        : 
        output        : tensor [bs*N,out_dim]/[bs,N,out_dim]
        description   : 
        """
        
        for layer_iter,layer in enumerate(self.layers):
            if layer_iter==0:
                output = layer(torch.cat((item_embedding,uv),dim=-1))
            elif layer_iter==3*(self.num_of_layers-1):
                output = layer(output)
            elif layer_iter%3==0:
                output = layer(torch.cat((output,uv),dim=-1))
            else:
                output = layer(output)       
        return output





    
