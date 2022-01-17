import logging
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, transformer_dim=64, n_head=8, dim_fc=128, n_layers=6, n_embedding_layers=4, use_pos_embedding=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.transformer_dim = transformer_dim if n_embedding_layers > 0 else input_dim
        self.n_head = get_nhead(self.transformer_dim, n_head)
        self.dim_fc = dim_fc
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers
        self.use_pos_embedding = use_pos_embedding
        
        self.input_proj = nn.ModuleList()
        for _ in range(self.n_embedding_layers):
            d_in = self.input_dim if len(self.input_proj) == 0 else self.transformer_dim
            conv_layer = nn.Sequential(nn.Conv1d(d_in, self.transformer_dim, 1), nn.GELU())
            self.input_proj.append(conv_layer)
            
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)))
        
        if self.use_pos_embedding:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
    
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = self.n_head,
                                       dim_feedforward = self.dim_fc,
                                       activation = 'gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = self.n_layers,
                                              norm = nn.LayerNorm(self.transformer_dim))
                                              

        self.imu_head = nn.Sequential(
                nn.LayerNorm(self.transformer_dim),
                nn.Linear(self.transformer_dim,  self.transformer_dim//4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.transformer_dim//4, output_dim))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        return 'Transformer_Encoder with dim={}'.format(self.transformer_dim)

    def forward(self, src):
        # Input [B, Win, D]
        # Reshaping to [B, D, Win]
        target = src.transpose(1, 2)
        
        # Embedding Input into (higher dim) latent vector-space -> [B, D', Win]
        for conv_layer in self.input_proj:
            target = conv_layer(target)
        
            
        # Reshaping: [B, D', Win] -> [Win, B, D'] 
        target = target.permute(2, 0, 1)
            
        # Prepend class token: [Win, B, D']  -> [Win+1, B, D']
        cls_token = self.cls_token.unsqueeze(1).repeat(1, target.shape[1], 1)
        target = torch.cat([cls_token, target])
    
        # Add the position embedding
        if self.use_pos_embedding:
            target += self.position_embed
        
        # Transformer Encoder pass
        target = self.transformer_encoder(target)[0]

        # Pass through fully-connected layers
        return self.imu_head(target)
    

def get_nhead(embed_dim, n_head):
    for hd in range(n_head, 0, -1):
        if embed_dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd