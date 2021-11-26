from .IMUCLSBaseline import IMUCLSBaseline
from .IMUTransformerEncoder import IMUTransformerEncoder

def get_model(model_name, input_dim, output_size, window_size, n_classes, model_config): 
    if model_name=='transformer':
        return IMUTransformerEncoder(input_dim, output_size, window_size, n_classes, model_config.get('transformer'))

    if model_name=='baseline':
        return IMUCLSBaseline(input_dim, output_size, window_size, n_classes, model_config.get('baseline'))
