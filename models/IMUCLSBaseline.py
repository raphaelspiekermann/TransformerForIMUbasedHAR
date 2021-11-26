import torch.nn as nn


class IMUCLSBaseline(nn.Module):
    def __init__(self, input_dim, output_size, window_size, n_classes, baseline_config):

        super(IMUCLSBaseline, self).__init__()

        feature_dim = baseline_config.get("dim")

        self.output_size = output_size

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, feature_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(baseline_config.get("dropout"))
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(window_size*(feature_dim//2), feature_dim, nn.ReLU())
        self.fc2 = nn.Linear(feature_dim,  n_classes) if output_size == 1 else nn.Linear(feature_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.n_classes = n_classes

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        x = data.get('imu').transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x) # return B X C/2 x M
        x = x.view(x.size(0), -1) # B X C/2*M
        x = self.fc1(x)
        x = self.log_softmax(self.fc2(x)) if self.output_size in [1, self.n_classes] else self.sigmoid(self.fc2(x))
        return x # B X N