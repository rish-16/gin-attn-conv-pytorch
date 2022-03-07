import os.path as osp
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from gin_attn_pytorch import GINAttnConv
from torch_geometric.datasets import CoraFull
import torch_geometric.transforms as T
import torch_geometric.nn as tgnn
from torch_geometric.datasets import Planetoid

dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
x, edge_index = data.x, data.edge_index
print (x.shape)

class GINAttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = GINAttnConv(dataset.num_features, 512, 64)
        self.g2 = GINAttnConv(512, 256, 64)
        self.g3 = GINAttnConv(256, dataset.num_classes, 32)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = tgnn.GATConv(dataset.num_features, 512)
        self.g2 = tgnn.GATConv(512, 256)
        self.g3 = tgnn.GATConv(256, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)

        return F.log_softmax(x, dim=1)     

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = tgnn.GCNConv(dataset.num_features, 512)
        self.g2 = tgnn.GCNConv(512, 256)
        self.g3 = tgnn.GCNConv(256, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)

        return F.log_softmax(x, dim=1)           

ginattn = GINAttnNet()
gat = GAT()
gcn = GCN()

ginattn_optim = torch.optim.Adam(ginattn.parameters(), lr=0.01, weight_decay=5e-3)
gat_optim = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=5e-3)
gcn_optim = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-3)

def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index)
    F.nll_loss(pred[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test(model):
    model.eval()
    log_probs, accs = model(x, edge_index), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

gat_train_acc = []
ginattn_train_acc = []
gcn_train_acc = []

for epoch in range(50):
    train(gat, gat_optim)
    train_acc, test_acc = test(gat)
    gat_train_acc.append(test_acc)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

for epoch in range(50):
    train(gcn, gcn_optim)
    train_acc, test_acc = test(gcn)
    gcn_train_acc.append(test_acc)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')        

for epoch in range(50):
    train(ginattn, ginattn_optim)
    train_acc, test_acc = test(ginattn)
    ginattn_train_acc.append(test_acc)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')            

plt.plot(range(50), gcn_train_acc, label="GCN", color="red")
plt.plot(range(50), gat_train_acc, label="GAT", color="blue")
plt.plot(range(50), ginattn_train_acc, label="GINAttn", color="green")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Testing Accuracy")
plt.title("Planetoid (Cora)")
plt.show()