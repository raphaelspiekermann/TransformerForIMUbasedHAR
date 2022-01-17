import torch.nn as nn
import numpy

class TCNN_2D(nn.Module):
    def __init__(self, input_dim, output_dim, win_size, imu_dim, n_filters=64, kernel_size=5, n_convolutions=2, fc_dim = 128, dropout=0.1, pooling_layer='None'):        
        super(TCNN_2D, self).__init__()
        assert input_dim % imu_dim == 0
        
        pooling_layer = pooling_layer.lower()
        assert pooling_layer in ['none', 'avg', 'max']
         
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.win_size = win_size
        self.imu_dim = imu_dim
        self.n_devices = self.input_dim // self.imu_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_convolutions = n_convolutions
        self.fc_dim = fc_dim
        self.use_pooling = pooling_layer in ['avg', 'max']

        self.convolutions = nn.ModuleList()
        for _ in range(self.n_convolutions):
            d_in = self.imu_dim if len(self.convolutions) == 0 else self.n_filters
            conv_layer = nn.Sequential(nn.Conv2d(d_in, self.n_filters, kernel_size=(self.kernel_size, 1)), nn.ReLU())
            self.convolutions.append(conv_layer)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.use_pooling:
            pooling_kernel = (1, self.n_devices)
            self.pooling_layer = nn.AvgPool2d(pooling_kernel) if pooling_layer=='avg' else nn.MaxPool2d(pooling_kernel)

        # Computing input_dim for the first fc-layer = filter * devices * win_size (after convolutions)
        win_size_after_conv = self.win_size - self.n_convolutions * (self.kernel_size - 1)
        d_in = self.n_filters * self.n_devices * win_size_after_conv
        d_in = d_in // self.n_devices if self.use_pooling else d_in
        
        self.fc1 = nn.Linear(d_in, self.fc_dim, nn.ReLU())
        self.fc2 = nn.Linear(self.fc_dim, output_dim)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def __str__(self):
        return 'TCNN_2D'

    def forward(self, data):
        # Input: [B, Win, In_Channel]
        # Reshaping to [batch_size, In_Channel, Win]
        target = data.transpose(1,2)
        
        # Reshaping to [B, IMU_Dim, N_IMU, Win]. For example [128, 30, 100] -> [128, 6, 5, 100] with: 128 batch-elements, 6 channels per IMU, 5 IMUs and a sliding window of size 100
        target = target.reshape((target.shape[0], self.imu_dim, self.n_devices, self.win_size))
        
        # Reshaping to [B, IMU_Dim, Win, N_IMU]
        target = target.transpose(2,3)
        
        # Applying convolutions
        for conv_layer in self.convolutions:
            target = conv_layer(target)
        
        # Dropout
        target = self.dropout(target)
        
        # Max- or Avg- Pooling if use_pooling is True -> new Shape [B, IMU_Dim, Win, 1]
        if self.use_pooling:
            target = self.pooling_layer(target)
        
        # Flattening the tensor to [B, x] for final fully connected layers.
        target = target.view(target.size(0), -1)

        target = self.fc1(target)
        return self.fc2(target)
