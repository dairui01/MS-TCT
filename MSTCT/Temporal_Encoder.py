from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn as nn

class Local_Relational_Block(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.TC = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features) # k=3, stride=1, padding=1
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.TC(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class Global_Relational_Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = None or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class GLRBlock(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.Global_Relational_Block = Global_Relational_Block(
            dim,num_heads=num_heads)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Local_Relational_Block = Local_Relational_Block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.Global_Relational_Block(self.norm1(x))
        x = x + self.Local_Relational_Block(self.norm2(x))
        return x


class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size// 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[256, 384, 576, 864],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=3):
        super().__init__()

        # Stage 1
        self.Temporal_Merging_Block1 = Temporal_Merging_Block(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                              embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([GLRBlock(
            dim=embed_dims[0], num_heads=num_head, mlp_ratio=mlp_ratio,norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage 2
        self.Temporal_Merging_Block2 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([GLRBlock(
            dim=embed_dims[1], num_heads=num_head, mlp_ratio=mlp_ratio,norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage 3
        self.Temporal_Merging_Block3 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([GLRBlock(
            dim=embed_dims[2], num_heads=num_head, mlp_ratio=mlp_ratio,norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage 4
        self.Temporal_Merging_Block4 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([GLRBlock(
            dim=embed_dims[3], num_heads=num_head, mlp_ratio=mlp_ratio,norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_init_emb(self):
        self.Temporal_Merging_Block1.requires_grad = False

    def forward(self, x):
        outs = []
        # stage 1
        x = self.Temporal_Merging_Block1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 2
        x = self.Temporal_Merging_Block2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 3
        x = self.Temporal_Merging_Block3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 4
        x = self.Temporal_Merging_Block4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        return outs