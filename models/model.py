from .IMU_Transformer_Encoder import IMUTransformerEncoder
from .Raw_IMU_Transformer_Encoder import RawIMUTransformerEncoder
from .IMU_CLS_Baseline import IMUCLSBaseline
from .IMU_LSTM import IMU_LSTM
from .LARa_TCNN import Lara_TCNN
import logging


def get_model(model_name, input_dim, output_size, window_size):
    # CNNs
    if model_name=='baseline':
        return IMUCLSBaseline(input_dim, output_size, window_size)
    if model_name in ['lara', 'lara_tcnn', 'lara_cnn', 'lara_baseline']:
        return Lara_TCNN(input_dim, output_size, 6, window_size)

    # Transformers
    if model_name.lower() in ['small', 'small_transformer']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, 32, 4, 64, 4)
    if model_name.lower() in ['medium', 'transformer', 'medium_transformer']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, 64, 8, 128, 6)
    if model_name.lower() in ['big', 'big_transformer']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, 128, 8, 256, 6)
    if model_name.lower() in ['raw', 'raw_transformer']:
        return RawIMUTransformerEncoder(input_dim, output_size, window_size, get_nhead(input_dim), 128, 6)

    # RNNs
    if model_name.lower()=='lstm':
        return IMU_LSTM(input_dim, output_size, window_size, 128)


def get_nhead(dim):
    for hd in range(min(8, dim-1), 0, -1):
        if dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd