import torch.nn as nn
import numpy

class Lara_TCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_channel, win_size):        
        super(Lara_TCNN, self).__init__()
        assert input_dim % n_channel == 0
         
        self.dim = input_dim // n_channel
        self.n_channel = n_channel
        self.win_size = win_size

        kernel_size = (5,1)

        self.conv1 = nn.Sequential(nn.Conv2d(n_channel, 64, kernel_size=kernel_size), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=kernel_size), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=kernel_size), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=kernel_size), nn.ReLU())     
        
        kernel_losses = 4 * (kernel_size[0] - 1)
        fc_in_dim =  64 * (self.win_size - kernel_losses) * self.dim

        self.fc1 = nn.Linear(fc_in_dim, 128, nn.ReLU())
        self.fc2 = nn.Linear(128, 128, nn.ReLU())
        self.fc3 = nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(0.1)
        self.avgpool = nn.AvgPool2d(kernel_size=[1, n_channel])

    def __str__(self):
        return 'LARa_TCNN'

    def forward(self, data):
        #data.shape = [batch_size, win_size, 30]
        target = data.transpose(1,2)  # new shape = [batch_size, 30, win_size]
        
        batch_size = target.shape[0]
        n_channel = self.n_channel
        dim = self.dim
        win_size = target.shape[2]
        target = target.reshape((batch_size, n_channel, dim, win_size))

        target = target.transpose(2,3)

        target = self.conv1(target)
        target = self.conv2(target)
        target = self.conv3(target)
        target = self.conv4(target)
        target = self.dropout(target)

        # reshaping target
        target = target.reshape((batch_size, numpy.prod(target.shape[1:])))

        target = self.fc1(target)
        target = self.fc2(target)
        return self.fc3(target)
