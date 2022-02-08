import torch
import torch.nn as nn


class IMU_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, lstm_dim=128, n_layers=1, n_embedding_layers=4, use_pos_embedding=True, use_class_token=False, activation_function='gelu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.lstm_dim = lstm_dim if n_embedding_layers > 0 else input_dim
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers
        self.use_pos_embedding = use_pos_embedding
        self.use_class_token = use_class_token
        self.activation_function = nn.GELU() if activation_function.lower() == 'gelu' else nn.ReLU()
        
        self.input_proj = nn.ModuleList()
        for _ in range(self.n_embedding_layers):
            d_in = self.input_dim if len(self.input_proj) == 0 else self.lstm_dim
            conv_layer = nn.Sequential(nn.Conv1d(d_in, self.lstm_dim, 1), self.activation_function)
            self.input_proj.append(conv_layer)
        
        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros((1, self.lstm_dim)))
        
        if self.use_pos_embedding:
            embedding_size = self.window_size + 1 if self.use_class_token else self.window_size
            self.position_embed = nn.Parameter(torch.randn(embedding_size, 1, self.lstm_dim))
        
        lstm_dropout = 0.1 if self.n_layers > 1 else 0
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, dropout=lstm_dropout, num_layers=self.n_layers)
        
        # The linear layer that maps from hidden state space to tag space
        self.imu_head = nn.Sequential(
                nn.LayerNorm(self.lstm_dim),
                nn.Linear(self.lstm_dim, self.lstm_dim // 4),
                self.activation_function,
                nn.Dropout(0.1),
                nn.Linear(self.lstm_dim // 4, output_dim))
    
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        return 'IMU_LSTM with activation {}'.format(str(self.activation_function))

    def forward(self, src):        
        # Input [B, Win, D]
        # Reshaping to [B, D, Win]
        target = src.transpose(1, 2)
        
        # Embedding Input into (higher dim) latent vector-space -> [B, D', Win]
        for conv_layer in self.input_proj:
            target = conv_layer(target)
            
        # Reshaping: [B, D', Win] -> [Win, B, D']
        target = target.permute(2, 0, 1)
        
        # Inserting class_token if used for classification
        if self.use_class_token:
            # Prepend class token: [Win, B, D']  -> [Win+1, B, D']
            cls_token = self.cls_token.unsqueeze(1).repeat(1, target.shape[1], 1)
            target = torch.cat([target, cls_token])
        
        # Add the position embedding
        if self.use_pos_embedding:
            target += self.position_embed

        # LSTM pass
        output, (hidden_out, _) = self.lstm(target)
        target = output[-1] if self.use_class_token else hidden_out[-1]
            
        # Pass through fully-connected layers
        return self.imu_head(target)