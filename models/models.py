## Standard libraries

## Imports for plotting

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()


## PyTorch

import torch.nn as nn
import torch.nn.functional as F

# Torchvision
import torch_geometric.nn as geom_nn
from torch_geometric.nn.norm import BatchNorm


class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class GATModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATConv
        self.conv1 = GAT(c_in, c_hidden, heads=heads1, edge_dim=edge_dim)
        self.bn1 = BatchNorm(c_hidden*heads1)
        self.conv2 = GAT(c_hidden*heads1, c_hidden, heads=heads2, edge_dim=edge_dim)
        self.bn2 = BatchNorm(c_hidden*heads2)
        self.conv3 = GAT(c_hidden*heads2, c_hidden, heads=heads3, edge_dim=edge_dim, concat=False)
        self.bn3 = BatchNorm(c_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """

        x, edge_index, batch_idx, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        x1 = self.bn1(self.conv1(x, edge_index, edge_weights))
        nn.ELU(x1, inplace=True)
        x2 = self.bn2(self.conv2(x1, edge_index, edge_weights)) + x1  # skip connection
        nn.ELU(x2, inplace=True)
        x3 = self.bn3(self.conv3(x2, edge_index, edge_weights))
        x3 = geom_nn.global_mean_pool(x3, batch_idx) # aggregate average pooling
        x4 = self.head(x3)

        return x4


class ConvModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate=0.1, lstm_steps=10, num_layers=2, **kwargs):
        """
        Model from https://arxiv.org/pdf/1905.06515.pdf for ncRNA graph classification
        Code inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        Conv = geom_nn.GCNConv
        self.conv1 = Conv(c_in, c_hidden)
        self.bn1 = BatchNorm(c_hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(Conv(c_hidden, c_hidden))
            self.bns.append(BatchNorm(c_hidden))
        self.set2set = geom_nn.Set2Set(c_hidden, lstm_steps)
        self.lin1 = nn.Linear(2 * c_hidden, c_hidden) # set2set doubles dims
        self.lin2 = nn.Linear(c_hidden, c_out)


    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """

        x, edge_index, batch_idx, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index, edge_weights)))
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(bn((conv(x, edge_index, edge_weights))) + x) # added skip connection
        x = self.set2set(x, batch_idx)
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return x

class GATConvModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, heads=10, dp_rate=0.1, lstm_steps=10, num_layers=2, edge_dim=1, **kwargs):
        """
        Model from https://arxiv.org/pdf/1905.06515.pdf for ncRNA graph classification
        Code inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.dp_rate = dp_rate
        GAT = geom_nn.GATConv
        self.gat1 = GAT(c_in, c_hidden, heads=heads, edge_dim=edge_dim)
        self.bn1 = BatchNorm(c_hidden*heads)
        self.gat2 = GAT(c_hidden*heads, c_hidden, heads=heads, edge_dim=edge_dim, concat=False)
        self.bn2 = BatchNorm(c_hidden)
        Conv = geom_nn.GCNConv
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(Conv(c_hidden, c_hidden))
            self.bns.append(BatchNorm(c_hidden))
        self.set2set = geom_nn.Set2Set(c_hidden, lstm_steps)
        self.lin1 = nn.Linear(2 * c_hidden, c_hidden) # set2set doubles dims
        self.lin2 = nn.Linear(c_hidden, c_out)

    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """

        x, edge_index, batch_idx, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.bn1(self.gat1(x, edge_index, edge_weights))
        nn.ELU(x, inplace=True)
        x = F.dropout(x, p=self.dp_rate, training=self.training)
        x = self.bn2(self.gat2(x, edge_index, edge_weights))
        nn.ELU(x, inplace=True)
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(bn((conv(x, edge_index, edge_weights))) + x) # added skip connection
        x = self.set2set(x, batch_idx)
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

class ConvSingleNodeModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate=0.1, lstm_steps=10, num_layers=2, **kwargs):
        """
        Model from https://arxiv.org/pdf/1905.06515.pdf for ncRNA graph classification
        Code inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        Conv = geom_nn.GCNConv
        self.conv1 = Conv(c_in, c_hidden)
        self.bn1 = BatchNorm(c_hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(Conv(c_hidden, c_hidden))
            self.bns.append(BatchNorm(c_hidden))
        self.lin1 = nn.Linear(c_hidden, c_hidden) # set2set doubles dims
        self.lin2 = nn.Linear(c_hidden, c_out)


    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """

        x, edge_index, batch_idx, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index, edge_weights)))
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(bn((conv(x, edge_index, edge_weights))) + x) # added skip connection
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return x

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers -1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )


    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        # mg_idx_mask = [np.where(batch_idx.cpu()==item)[0][0] for item in range(max(batch_idx)+1)] # classify only the 0th node in every graph
        # x = x[mg_idx_mask, :]
        #TODO for single node classification uncomment above and get rid of pooling.
        x = self.head(x)
        return x

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}