from .Transformer import IMUTransformerEncoder
from .IMU_CLS_Baseline import IMUCLSBaseline
from .IMU_LSTM import IMU_LSTM
from .LARa_TCNN import Lara_TCNN


def get_model(model_name, dim_in, dim_out, window_size):
    model_name = model_name.lower()
    
    # CNNs
    if model_name == 'baseline':
        return IMUCLSBaseline(dim_in, dim_out, window_size)
    if model_name == 'lara_cnn':
        return Lara_TCNN(dim_in, dim_out, 6, window_size)

    # Transformers
    if model_name in ['transformer', 'small', 'raw', 'lara_transformer']:
        return IMUTransformerEncoder(dim_in, dim_out, window_size, model_name)
    
    # RNNs
    if model_name == 'lstm':
        return IMU_LSTM(dim_in, dim_out, window_size, 64)
    
    raise RuntimeError