import torch
import torch.nn as nn 
from torch_geometric.nn import MLP, PointNetConv, radius, global_max_pool, fps
from torch_geometric.data import Batch
from classification_latents.layers import build_mlp



class ClassificationModel(nn.Module):
    def __init__(
            self
    ):
        super(ClassificationModel, self).__init__()

        self.mlp1 = MLP([8, 16, 8, 1], act="ReLU", dropout=0.4)

        self.mlp2 = MLP([1024, 256, 64, 16, 2], act="ReLU", dropout=0.4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph: Batch):

        x = graph.x

        x = self.mlp1(x)  
        
        if graph.num_graphs > 1:
            x = x.view(graph.num_graphs, -1) 
        else:
            x = x.view(1, -1)

        x = self.mlp2(x)  
        
        return self.softmax(x)
    


# The PointNet++ classification model and layer
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, number_of_connections=16):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.number_of_connections = number_of_connections

    def _radius(self, idx, pos, batch):
        return radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.number_of_connections,
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = self._radius(idx, pos, batch)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, pos, batch):
        x = self.net(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ClassificationPointNetP2(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 8,
        dim_model: list = [
            [64, 128, 128],
            [128, 128, 256],
            [256, 512, 1024],
            [1024, 512]
        ],
        output_size: int = 2,
        number_of_connections: int = 16,
        **kwargs
    ):
        super().__init__()

        self.sa_modules = nn.ModuleList()
        # Initialize the first SAModule
        self.sa_modules.append(
            SAModule(
                0.5,
                0.2,
                build_mlp(
                    3 + node_input_size,
                    dim_model[0][0],
                    dim_model[0][-1],
                    len(dim_model[0]),
                ),
                number_of_connections,
            )
        )

        # Add the intermediate SAModules
        for i in range(1, len(dim_model) - 2):
            self.sa_modules.append(
                SAModule(
                    0.25,
                    0.4,
                    build_mlp(
                        dim_model[i - 1][-1] + 3,
                        dim_model[i][0],
                        dim_model[i][-1],
                        len(dim_model[i]),
                    ),
                    number_of_connections,
                )
            )

        # Add the final GlobalSAModule
        self.sa_modules.append(
            GlobalSAModule(
                build_mlp(
                    dim_model[-3][-1] + 3,
                    dim_model[-2][0],
                    dim_model[-2][-1],
                    len(dim_model[-1]),
                )
            )
        )


        self.mlp = build_mlp(
            dim_model[-1][0],
            dim_model[-1][1],
            output_size,
            2,
            dropout=0.5,
            layer_norm=False,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)

        for sa in self.sa_modules:
            sa0_out = sa(*sa0_out)

        x, pos, batch = sa0_out

        x = self.mlp(x)

        return self.softmax(x)

        # return self.mlp(x).log_softmax(dim=-1)


