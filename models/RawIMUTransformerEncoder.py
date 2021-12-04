import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RawIMUTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, n_classes,  
                                    n_head, dim_feed_forward, n_layers, encode_position):
        super().__init__()

        self.transformer_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.encode_position = encode_position
        self.n_classes = n_classes
 
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = n_head,
                                       dim_feedforward = dim_feed_forward,
                                       dropout = 0.1,
                                       activation = 'gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = n_layers,
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

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
        return 'Transformer_Encoder with dim={}.'.format(self.transformer_dim)

    def forward(self, src):
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = src.transpose(1, 2).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed
        
        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
    
        # Class/Attr probability
        return self.imu_head(target)
