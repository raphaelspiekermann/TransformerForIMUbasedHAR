import torch
import torch.nn as nn

class IMU_RNN(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, n_classes, rnn_dim, encode_position):
        super().__init__()
        self.output_dim = output_dim
        self.window_size = window_size
        self.encode_position = encode_position
        self.n_classes = n_classes
        
        self.input_proj = nn.Sequential(nn.Conv1d(input_dim, rnn_dim, 1), nn.GELU(),
                                nn.Conv1d(rnn_dim, rnn_dim, 1), nn.GELU(),
                                nn.Conv1d(rnn_dim, rnn_dim, 1), nn.GELU(),
                                nn.Conv1d(rnn_dim, rnn_dim, 1), nn.GELU())
 
        self.position_embed = nn.Parameter(torch.randn(window_size+1, 1, rnn_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, rnn_dim)), requires_grad=True)

        self.rnn = nn.RNN(rnn_dim, rnn_dim, dropout=0.1, num_layers=6)
        
        # The linear layer that maps from hidden state space to tag space
        self.imu_head = nn.Sequential(
            nn.LayerNorm(rnn_dim),
            nn.Linear(rnn_dim,  rnn_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(rnn_dim//4, output_dim))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # Embed in a high dimensional space and reshape to LSTM's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed
        
        # Transformer Encoder pass
        target, _ = self.rnn(src)
        target = target[0]
    
        # Class/Attr probability
        return self.imu_head(target)