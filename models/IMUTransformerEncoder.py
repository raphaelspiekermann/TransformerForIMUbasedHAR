"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, input_dim, output_size, window_size, n_classes, transformer_config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = transformer_config.get("dim")
        
        self.output_size = output_size

        self.input_proj = nn.Sequential(nn.Conv1d(input_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = window_size
        self.encode_position = transformer_config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = transformer_config.get("nhead"),
                                       dim_feedforward = transformer_config.get("dim_feedforward"),
                                       dropout = transformer_config.get("dropout"),
                                       activation = transformer_config.get("activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = transformer_config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        if output_size == 1:
            self.imu_head = nn.Sequential(
                nn.LayerNorm(self.transformer_dim),
                nn.Linear(self.transformer_dim,  self.transformer_dim//4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.transformer_dim//4,  n_classes)
            )
        else:              
            self.imu_head = nn.Sequential(
                nn.LayerNorm(self.transformer_dim),
                nn.Linear(self.transformer_dim,  self.transformer_dim//4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.transformer_dim//4, output_size)
            )


        self.log_softmax = nn.LogSoftmax(dim=1) 
        self.sigmoid = nn.Sigmoid()
        self.n_classes = n_classes

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)


        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed
    
        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
   
        # Class probability
        #TODO one-hot-encoding schöner einbauen wenn Zeit dafür
        target = self.log_softmax(self.imu_head(target)) if self.output_size in [1, self.n_classes] else self.sigmoid(self.imu_head(target))
        return target
