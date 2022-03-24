import torch.nn as nn

class Classification_Module(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.linear_fuse = nn.Conv1d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1)
        self.linear_pred = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)

        self.hm = nn.Sequential(
            nn.Conv1d(embedding_dim * 4, embedding_dim, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.dropout = nn.Dropout()

    def forward(self, concat_feature, concat_feature_hm):
        # Heat-map Branch
        x_hm = self.hm(concat_feature_hm)

        # Classification Branch
        x = self.linear_fuse(concat_feature)
        x = self.dropout(x)
        x = self.linear_pred(x)
        x = x.permute(0, 2, 1)

        return x, x_hm
