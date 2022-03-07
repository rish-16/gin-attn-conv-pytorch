from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tgnn

# class Attention(nn.Module):
#     def __init__(self, features, attn_dim):
#         super(Attention, self).__init__()
#         self.to_q = nn.Linear(features, attn_dim)
#         self.to_k = nn.Linear(features, attn_dim)
#         self.to_v = nn.Linear(features, attn_dim)
#         self.project = nn.Linear(attn_dim, features)
        
#     def forward(self, x):
#         Q = self.to_q(x)
#         K = self.to_k(x)
#         V = self.to_v(x)
        
#         dots = torch.bmm(Q, K.permute(0, 2, 1))
#         attn = F.softmax(dots, 0)
        
#         out = torch.bmm(attn, V)
#         out = self.project(out)
        
#         return out

class GINAttnConv(tgnn.MessagePassing):
    def __init__(self, indim, outdim, attn_dim):
        super().__init__(aggr="add")
        self.l1 = nn.Linear(indim, outdim)
        self.l2 = nn.Linear(indim, outdim)
        self.eps = 1e-2

        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.proj_back = torch.nn.Linear(2 * outdim, outdim)
        
        self.a = torch.nn.Parameter(torch.zeros(2 * outdim, 1))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_index):
        x = self.l1(x)
        proj = (1 + self.eps) * x
        return proj

    def message(self, x_i, x_j): # how to transform the neighbour feature
        proj_i = self.l1(x_i)
        proj_j = self.l2(x_j)
        
        cat = torch.cat([proj_i, proj_j], dim=1)
        tmp = self.a.T * cat
        out = F.softmax(self.leaky_relu(tmp), 1)
        out = self.proj_back(out)
        
        return out

        # return self.l2(x_j)