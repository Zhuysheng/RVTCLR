import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


class IntraPart(nn.Module):
    # part_5
    def __init__(self,hidden_dim):
        super().__init__()
        self.torso = [0, 1, 2, 3, 20]
        self.left_leg = [16, 17, 18, 19]
        self.right_leg = [12, 13, 14, 15]
        self.left_arm = [8, 9, 10, 11, 23, 24]
        self.right_arm = [4, 5, 6, 7, 21, 22]
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2,1,bias=False),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim//2, hidden_dim,1,bias=False))
    def forward(self, x):
        NM, C, T, V = x.shape
        torso = x[:, :, :, self.torso]
        left_leg = x[:, :, :, self.left_leg]
        right_leg = x[:, :, :, self.right_leg]
        left_arm = x[:, :, :, self.left_arm]
        right_arm = x[:, :, :, self.right_arm]

        x_torso = F.avg_pool2d(torso, kernel_size=(1, 5))  # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(left_leg, kernel_size=(1, 4))  # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(right_leg, kernel_size=(1, 4))  # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(left_arm, kernel_size=(1, 6))  # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(right_arm, kernel_size=(1, 6))  # [N, C, T, V=1]
        x_avg = torch.cat((x_torso,x_leftleg,x_rightleg,x_leftarm,x_rightarm),dim=-1).squeeze()
        x_avg_out = self.se(x_avg)

        out = x_avg_out
        #print(out.shape)
        return out


class InterPart(nn.Module): #[NM,C,T,5]
    def __init__(self, hidden_dim):
        super().__init__()
        self.inter_channels = hidden_dim // 2
        self.g = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2,
                         kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2,
                             kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim)
            )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self,x):

        N, C, T, V = x.size()
        x = x.permute(0,3,1,2)
        x = x.reshape(N*V,C,T)
        g_x = self.g(x)
        g_x = g_x.view(N, V*self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        xmean = x.view(N, V, C, T).mean(dim=1)
        theta_x = self.theta(xmean).view(N, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(xmean).view(N, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(N * V, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1)
        return z

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, no_pretrain=False, **kwargs):
        super().__init__()
        self.no_pretrain = no_pretrain
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        self.fc1 = nn.Linear(hidden_dim, num_class)
        self.fc2 = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.part = IntraPart(hidden_dim)
        self.relation = InterPart(hidden_dim)

    def forward(self, x, DC=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # prediction
        if self.no_pretrain:
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)
            x = self.fc(x)    #close if do tsne
            x = x.view(x.size(0), -1)
            return x
        else:
            if not DC:
                x = F.avg_pool2d(x, x.size()[2:])
                x = x.view(N, M, -1).mean(dim=1)
                x1 = self.fc1(x)
                x2 = self.fc2(x)
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.view(x2.size(0), -1)
                return x1, x2
            else:
                x = self.part(x)  # [NM,C,T,5] #inter-part
                x = self.relation(x) #[NM,C,T,5]
                x = F.avg_pool2d(x, x.size()[2:])
                x = x.view(N, M, -1).mean(dim=1)
                x = self.fc(x)
                x = x.view(x.size(0), -1)
                return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A




