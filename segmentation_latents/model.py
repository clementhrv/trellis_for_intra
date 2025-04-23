import torch
import torch.nn as nn 
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import MLP, PointNetConv, radius, global_max_pool, fps, MessagePassing, knn_interpolate
from torch_geometric.data import Batch
from segmentation_latents.layers import build_mlp


# The PointNet classification model and layer with message passing
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = build_mlp(
            in_size=in_channels + 3,
            hidden_size=out_channels,
            out_size=out_channels,
            nb_of_layers=2,
            layer_norm=False,
        )

    def forward(
        self,
        graph: Batch,
    ) -> Tensor:
        # Start propagating messages.
        x = graph.x
        edge_index = graph.edge_index
        pos = graph.pos
        return self.propagate(edge_index, h=x, pos=pos)

    def message(
        self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNetClassifier(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 8,
        hidden_layers: int = 2,
        hidden_size: int = 64,
        output_size: int = 2,
        **kwargs
    ):
        super().__init__()

        self.processer_list = nn.ModuleList(
            [PointNetLayer(node_input_size, hidden_size), ReLU()]
        )

        for _ in range(hidden_layers):
            self.processer_list.append(PointNetLayer(hidden_size, hidden_size))
            self.processer_list.append(ReLU())

        self.classifier = Linear(hidden_size, output_size)

    def forward(self, graph=Batch) -> Tensor:

        for layer in self.processer_list:
            graph.x = layer(graph)

        # Global Pooling:
        x = global_max_pool(graph.x, graph.batch)  # [num_examples, hidden_channels]

        x = self.classifier(x)  # [num_examples, output_channels]

        # Classifier:
        return x.log_softmax(dim=1)

# The PointNet++ segmentation model and layer
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


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos, batch


class SegmentationPointNetP2(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 8,
        dim_model: list = [
            [[64, 128, 128],
            [128, 128, 256],
            [256, 512, 1024]],
            [[256, 256, 256],
            [128, 256, 128],
            [128, 128, 128],],
            [128, 128, 128]
        ],
        output_size: int = 3,
        number_of_connections: int = 16,
        **kwargs
    ):
        super().__init__()

        self.sa_modules = nn.ModuleList()
        # Initialize the first SAModule
        self.sa_modules.append(
            SAModule(
                0.2,
                0.2,
                build_mlp(
                    3 + node_input_size,
                    dim_model[0][0][0],
                    dim_model[0][0][-1],
                    len(dim_model[0][0]),
                ),
                number_of_connections,
            )
        )

        # Add the intermediate SAModules
        for i in range(1, len(dim_model[0]) - 1):
            self.sa_modules.append(
                SAModule(
                    0.25,
                    0.4,
                    build_mlp(
                        dim_model[0][i - 1][-1] + 3,
                        dim_model[0][i][1],
                        dim_model[0][i][-1],
                        len(dim_model[0][i]),
                    ),
                    number_of_connections,
                )
            )

        # Add the final GlobalSAModule
        self.sa_modules.append(
            GlobalSAModule(
                build_mlp(
                    dim_model[0][-2][-1] + 3,
                    dim_model[0][-1][1],
                    dim_model[0][-1][-1],
                    len(dim_model[0][-1]),
                )
            )
        )

        self.fp_modules = nn.ModuleList()

        self.fp_modules.append(
            FPModule(
                1,
                build_mlp(
                    dim_model[0][-1][-1] + dim_model[1][0][0],
                    dim_model[1][0][1],
                    dim_model[1][0][-1],
                    len(dim_model[1][0]),
                ),
            )
        )

        for i in range(1, len(dim_model[1]) - 1):
            self.fp_modules.append(
                FPModule(
                    3,
                    build_mlp(
                        dim_model[1][i - 1][-1] + dim_model[1][i][0],
                        dim_model[1][i][1],
                        dim_model[1][i][-1],
                        len(dim_model[1][i]),
                    ),
                )
            )

        self.fp_modules.append(
            FPModule(
                3,
                build_mlp(
                    dim_model[1][-2][-1] + node_input_size,
                    dim_model[1][-1][1],
                    dim_model[1][-1][-1],
                    len(dim_model[1][-1]),
                ),
            )
        )


        self.mlp = build_mlp(
            dim_model[-1][0],
            dim_model[-1][1],
            output_size,
            len(dim_model[-1])+1,
            dropout=0.5,
            layer_norm=False,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        sa0_out = [(data.x, data.pos, data.batch)]

        for i in range(len(self.sa_modules)):
            sa0_out.append(self.sa_modules[i](*sa0_out[i]))

        fp_out = [sa0_out[-1]]
        for i in range(len(self.fp_modules)):
            fp_out.append(self.fp_modules[i](*fp_out[i], *sa0_out[-2 - i]))

        x, pos, batch = fp_out[-1]

        x = self.mlp(x)

        return x.log_softmax(dim=1)


