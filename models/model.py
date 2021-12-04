from .IMUTransformerEncoder import IMUTransformerEncoder
from .IMUCLSBaseline import IMUCLSBaseline

def get_model(model_name, input_dim, output_size, window_size, n_classes, cfg): 
    if model_name=='baseline':
        return IMUCLSBaseline(input_dim, output_size, window_size, n_classes)

    encode_postion = cfg['encode_position']
    if model_name.lower() in ['small', 'small_transformer','small_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 32, 4, 64, 4, encode_postion)
    if model_name.lower() in ['medium', 'transformer', 'medium_transformer','medium_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 64, 8, 128, 6, encode_postion)
    if model_name.lower() in ['big', 'big_transformer','big_transformer_encoder']:
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, 512, 16, 1024, 8, encode_postion)