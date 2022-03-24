import torch
import torch.nn as nn
import torch.nn.functional as F


class linear_layer(nn.Module):
    #
    def __init__(self, input_dim=2048, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Temporal_Mixer(nn.Module):
    def __init__(self, inter_channels, embedding_dim):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = inter_channels

        self.linear_f4 = linear_layer(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_f3 = linear_layer(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_f2 = linear_layer(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_f1 = linear_layer(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear4 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

    def forward(self, x):
        f1, f2, f3, f4 = x

        # Temporal Scale Mixer Module
        _f4 = self.linear_f4(f4).permute(0,2,1)
        _f4 = resize(_f4, size=f1.size()[2:],mode='linear',align_corners=False)

        _f3 = self.linear_f3(f3).permute(0,2,1)
        _f3 = resize(_f3, size=f1.size()[2:],mode='linear',align_corners=False)

        _f2 = self.linear_f2(f2).permute(0,2,1)
        _f2 = resize(_f2, size=f1.size()[2:],mode='linear',align_corners=False)

        _f1 = self.linear_f1(f1).permute(0,2,1)

        # Mixer
        _f3_n = self.linear1(_f4) + _f3
        _f2_n = self.linear2(_f4) + _f2
        _f1_n = self.linear3(_f4) + _f1

        concat_feature=torch.cat([_f4, _f3_n, _f2_n, _f1_n], dim=1)
        concat_feature_hm = torch.cat([_f4, _f3, _f2, _f1], dim=1)

        return concat_feature, concat_feature_hm
