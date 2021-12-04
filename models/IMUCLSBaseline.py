import torch.nn as nn

class IMUCLSBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, n_classes):

        super(IMUCLSBaseline, self).__init__()

        feature_dim = 64

        self.output_dim = output_dim

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, feature_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(window_size*(feature_dim//2), feature_dim, nn.ReLU())
        self.fc2 = nn.Linear(feature_dim,  n_classes) if output_dim == 1 else nn.Linear(feature_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.n_classes = n_classes

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def __str__(self):
        return 'IMUCLSBaseline'

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        target = data.transpose(1, 2)
        target = self.conv1(target)
        target = self.conv2(target)
        target = self.dropout(target)
        target = self.maxpool(target) # return B X C/2 x M
        target = target.view(target.size(0), -1) # B X C/2*M
        target = self.fc1(target)
        return self.fc2(target)
        