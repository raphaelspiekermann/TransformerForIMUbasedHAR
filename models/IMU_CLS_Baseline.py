import torch.nn as nn

class IMUCLSBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, n_classes):
        super(IMUCLSBaseline, self).__init__()

        cnn_dim = 64
        self.output_dim = output_dim
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, cnn_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(cnn_dim, cnn_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(window_size*(cnn_dim//2), cnn_dim, nn.ReLU())
        self.fc2 = nn.Linear(cnn_dim, output_dim)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def __str__(self):
        return 'IMUCLSBaseline'

    def forward(self, data):
        print(data.shape)
        target = data.transpose(1, 2)
        print(target.shape)
        target = self.conv1(target)
        print(target.shape)
        target = self.conv2(target)
        print(target.shape)
        target = self.dropout(target)
        print(target.shape)
        target = self.maxpool(target) # return B X C/2 x M
        print(target.shape)
        target = target.view(target.size(0), -1) # B X C/2*M
        print(target.shape)
        target = self.fc1(target)
        return self.fc2(target)
        