import torch
import torch.nn as nn


class IMU_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, model_type):
        super().__init__()
        self.output_dim = output_dim
        self.window_size = window_size
        self.lstm_dim = 128
        self.n_layers = 2
        self.model_type = model_type
        
        self.input_proj = nn.Sequential(nn.Conv1d(input_dim, self.lstm_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.lstm_dim, self.lstm_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.lstm_dim, self.lstm_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.lstm_dim, self.lstm_dim, 1), nn.GELU())
        
        
        self.position_embed = nn.Parameter(torch.randn(self.window_size, 1, self.lstm_dim), requires_grad=True)
        
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, dropout=0.1, num_layers=self.n_layers)
        
        # The linear layer that maps from hidden state space to tag space
        self.imu_head = nn.Sequential(
                nn.LayerNorm(self.lstm_dim),
                nn.Linear(self.lstm_dim, self.lstm_dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.lstm_dim // 4, output_dim))
    
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        return 'IMU_LSTM'

    def forward(self, src):
        # Embed in a high dimensional space and reshape to LSTM's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)
        
        # Add the position embedding
        src += self.position_embed

        # LSTM pass
        _, (h_out, _) = self.lstm(src)
        target = h_out[-1]
            
        # Class/Attr probability
        return self.imu_head(target)