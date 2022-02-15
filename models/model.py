import json
from os.path import join, split, realpath
import models.Transformer as Transformer
import models.LSTM as LSTM
import models.TCNN_2D as TCNN_2D
import models.TCNN_1D as TCNN_1D

def get_model(model_name, dim_in, dim_out, window_size):
    
    path = split(realpath(__file__))[0]
    with open(join(path, 'model_configs.json')) as f:
        model_config = json.load(f)
    
    # 1D-CNNs
    if model_name in ['Baseline', 'tCNN-1D']:
        cfg = model_config[model_name]
        return TCNN_1D.TCNN_1D(dim_in, dim_out, window_size, cfg['n_filters'], cfg['kernel_size'], cfg['n_convolutions'], cfg['fc_dim'], cfg['dropout'], cfg['pooling_layer'])
    
    # 2D-CNNs
    if model_name in ['tCNN-2D']:
        cfg = model_config[model_name]
        return TCNN_2D.TCNN_2D(dim_in, dim_out, window_size, cfg['imu_dim'], cfg['n_filters'], cfg['kernel_size'], cfg['n_convolutions'], cfg['fc_dim'], cfg['dropout'], cfg['pooling_layer'])

    # Transformer
    if model_name in ['Transformer-Default', 'Transformer-Big', 'Transformer-NoPos', 'Transformer-NoCNN']:
        cfg = model_config[model_name]
        return Transformer.IMUTransformerEncoder(dim_in, dim_out, window_size, cfg['transformer_dim'], cfg['n_head'], cfg['fc_dim'], cfg['n_layers'], cfg['n_embedding_layers'], cfg['use_pos_embedding'], cfg['activation'])
    
    # RNNs
    if model_name in ['LSTM-Hidden', 'LSTM-Token']:
        cfg = model_config[model_name]
        return LSTM.IMU_LSTM(dim_in, dim_out, window_size, cfg['lstm_dim'], cfg['n_layers'], cfg['n_embedding_layers'], cfg['use_pos_embedding'], cfg['use_class_token'], cfg['activation'])
    
    raise RuntimeError('{} is not a known model'.format(model_name))