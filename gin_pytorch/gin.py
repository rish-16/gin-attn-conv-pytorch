import torch.nn as nn
import torch_geometric as tg
import torch_geometric.nn as tgnn

class GINConv(tgnn.MessagePassing):
    def __init__(self, indim, outdim):
        super().__init__(aggr="add")
        self.l1 = nn.Linear(indim, 128)
        self.l2 = nn.Linear(128, outdim)
        self.mlp = nn.Linear(indim, outdim)
        self.eps = 1e-2

    def forward(self, x, edge_idx):
        x = self.l1(x)
        x = self.l2(x)
        x = (1 + self.eps) * x

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(x_i + x_j)