import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder


class MSTCT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes):
        super(MSTCT, self).__init__()

        self.dropout=nn.Dropout()

        self.TemporalEncoder=TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block)

        self.Temporal_Mixer=Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        # Temporal Encoder Module
        x = self.TemporalEncoder(inputs)

        # Temporal Scale Mixer Module
        concat_feature, concat_feature_hm = self.Temporal_Mixer(x)

        # Classification Module
        x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

        return x, x_hm # B, T, C





