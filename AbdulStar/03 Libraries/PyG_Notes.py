# ========================================== 
# Installation:
# ========================================== 

# import torch

# # def format_pytorch_version(version):
# #   return version.split('+')[0]

# # TORCH_version = torch.__version__
# # TORCH = format_pytorch_version(TORCH_version)

# # def format_cuda_version(version):
# #   return 'cu' + version.replace('.', '')

# # CUDA_version = torch.version.cuda
# # CUDA = format_cuda_version(CUDA_version)

# # !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# # !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# # !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# # !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# # !pip install torch-geometric


# ========================================== 
# Intro:
# ========================================== 

# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html


# ----------------- 
# Data
# ----------------- 
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

data.keys

data['x']

data.num_nodes

data.num_edges

data.num_node_features

data.has_isolated_nodes()

data.has_self_loops()

data.is_directed()

# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)



# ----------------- 
# DataSets
# ----------------- 
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

len(dataset)

dataset.num_classes

dataset.num_node_features

data = dataset[0]

data.is_undirected()

data.train_mask.sum().item()

data.val_mask.sum().item()

data.test_mask.sum().item()


# ----------------- 
# Loader
# ----------------- 
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch in loader:
#     batch
#     >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])
#     batch.num_graphs
#     >>> 32


from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for data in loader:
#     data
#     >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

#     data.num_graphs
#     >>> 32

#     x = scatter_mean(data.x, data.batch, dim=0)
#     x.size()
#     >>> torch.Size([32, 21])


# ----------------- 
# Transforms
# ----------------- 
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(
                root='/tmp/ShapeNet', 
                categories=['Airplane'],
                pre_transform=T.KNNGraph(k=6)     # pre_transform
                transform=T.RandomTranslate(0.01) # transform
                )

dataset[0]


# ----------------- 
# Learning
# ----------------- 

# Import 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate 
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')



# ----------------- 
# Visualization:
# ----------------- 
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)

## OR
# model = GCN() 
# _, h = model(data.x, data.edge_index) # See the definition of GCN in colab 0 here:https://github.com/hdvvip/CS224W_Winter2021
# print(f'Embedding shape: {list(h.shape)}')
# visualize(h, color=data.y)