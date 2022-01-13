import logging
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def get_settings(m, input_dim):
    if m == 'small':
        return 32, 4, 64, 4
    if m == 'transformer':
        return 64, 8, 128, 6
    if m == 'raw':
        return input_dim, get_nhead(input_dim), 128, 6
    if m == 'lara_transformer':
        return 64, 8, 128, 6
    raise RuntimeError


class IMUTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, model_type):
        super().__init__()

        transformer_dim, n_head, dim_feed_forward, n_layers = get_settings(model_type, input_dim)
        
        self.transformer_dim = transformer_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.model_type = model_type
        
        if model_type == 'transformer':
            self.input_proj = nn.Sequential(nn.Conv1d(input_dim, self.transformer_dim, 1), nn.GELU(),
                                    nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                    nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                    nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())
        
        if model_type in ['small', 'lara_transformer']:
            self.input_proj = nn.Sequential(nn.Conv1d(input_dim, self.transformer_dim, 1), nn.GELU())
             
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = n_head,
                                       dim_feedforward = dim_feed_forward,
                                       dropout = 0.1,
                                       activation = 'gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = n_layers,
                                              norm = nn.LayerNorm(self.transformer_dim))
                                              
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim), requires_grad=True)

        if model_type in ['transformer', 'lara_transformer']:
            self.imu_head = nn.Sequential(
                nn.LayerNorm(self.transformer_dim),
                nn.Linear(self.transformer_dim,  self.transformer_dim//4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.transformer_dim//4, output_dim))
        else:
            self.imu_head = nn.Sequential(
                nn.LayerNorm(self.transformer_dim),
                nn.Linear(self.transformer_dim, output_dim))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        return 'Transformer_Encoder with dim={}'.format(self.transformer_dim)

    def forward(self, src):
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        if self.model_type != 'raw':
            src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)
        else:
            src = src.permute(1, 0, 2)
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        src += self.position_embed
        
        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # Class/Attr probability
        return self.imu_head(target)


def get_nhead(dim):
    for hd in range(min(8, dim-1), 0, -1):
        if dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd