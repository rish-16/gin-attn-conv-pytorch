import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import CoraFull
import torch_geometric.nn as tgnn
import torch_geometric.transforms as T
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

class Net(torch.nn.Module):
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

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index)
    F.nll_loss(pred[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(x, edge_index), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')