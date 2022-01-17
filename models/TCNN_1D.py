import torch.nn as nn

class TCNN_1D(nn.Module):
    def __init__(self, input_dim, output_dim, win_size, n_filters=64, kernel_size=1, n_convolutions=2, fc_dim=128, dropout=0.1, pooling_layer='None'):  
        super(TCNN_1D, self).__init__()
        
        pooling_layer = pooling_layer.lower()
        assert pooling_layer in ['none', 'avg', 'max']
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.win_size = win_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_convolutions = n_convolutions
        self.fc_dim = fc_dim
        self.use_pooling = pooling_layer in ['avg', 'max']
        
        self.convolutions = nn.ModuleList()
        for _ in range(self.n_convolutions):
            d_in = self.input_dim if len(self.convolutions) == 0 else self.n_filters
            conv_layer = nn.Sequential(nn.Conv1d(d_in, self.n_filters, kernel_size=self.kernel_size), nn.ReLU())
            self.convolutions.append(conv_layer)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.use_pooling:
            self.pooling_layer = nn.AvgPool1d(2) if pooling_layer=='avg' else nn.MaxPool1d(2)
        
        # Computing input_dim for the first fc-layer
        d_in = self.n_filters * (self.win_size - self.n_convolutions * (self.kernel_size - 1))
        d_in = d_in // 2 if self.use_pooling else d_in
        
        self.fc1 = nn.Linear(d_in, self.fc_dim, nn.ReLU())
        self.fc2 = nn.Linear(self.fc_dim, self.output_dim)
        
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

    def __str__(self):
        return 'TCNN_1D'

    def forward(self, data):
        # Input [B, Win, D]
        # Reshaping to [B, D, Win]
        target = data.transpose(1,2)
        
        # Applying the convolutions
        for conv_layer in self.convolutions:
            target = conv_layer(target)
        
        # Dropout
        target = self.dropout(target)
        
        # Max- or Avg- Pooling if use_pooling is True
        if self.use_pooling:
            target = self.pooling_layer(target)
        
        # Flattening the tensor to [B, x] for final fully connected layers.
        target = target.view(target.size(0), -1)

        # Applying 2 FC_Layers
        target = self.fc1(target)
        return self.fc2(target)
