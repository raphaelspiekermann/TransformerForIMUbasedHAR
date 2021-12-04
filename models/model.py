from .IMUTransformerEncoder import IMUTransformerEncoder
from .RawIMUTransformerEncoder import RawIMUTransformerEncoder
from .IMUCLSBaseline import IMUCLSBaseline
from .IMU_LSTM import IMU_LSTM
from .IMU_RNN import IMU_RNN
import logging


def get_model(model_name, input_dim, output_size, window_size, n_classes, cfg): 
    # CNNs
    if model_name=='baseline':
        return IMUCLSBaseline(input_dim, output_size, window_size, n_classes)

    # Transformers
    encode_postion = cfg['encode_position']
    if model_name.lower() in ['small', 'small_transformer','small_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 32, 4, 64, 4, encode_postion)
    if model_name.lower() in ['medium', 'transformer', 'medium_transformer','medium_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 64, 8, 128, 6, encode_postion)
    if model_name.lower() in ['big', 'big_transformer','big_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 144, 12, 256, 8, encode_postion)
    if model_name.lower() in ['raw', 'raw_transformer']:
        return RawIMUTransformerEncoder(input_dim, output_size, window_size, n_classes, get_nhead(input_dim), 128, 6, encode_postion)

    # RNNs
    if model_name.lower()=='lstm':
        return IMU_LSTM(input_dim, output_size, window_size, n_classes, 64, encode_postion)
    if model_name.lower()=='rnn':
        return IMU_RNN(input_dim, output_size, window_size, n_classes, 64, encode_postion)


def get_nhead(dim):
    for hd in range(min(8, dim-1), 0, -1):
        if dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd