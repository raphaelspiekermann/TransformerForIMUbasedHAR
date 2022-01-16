import models.Transformer as Transformer
import models.IMU_LSTM as LSTM
import models.IMU_CLS_Baseline as Baseline
import models.LARa_TCNN as TCNN

def get_model(model_name, dim_in, dim_out, window_size):
    model_name = model_name.lower()
    
    # CNNs
    if model_name in ['baseline', 'cnn']:
        return Baseline.IMUCLSBaseline(dim_in, dim_out, window_size)
    if model_name in ['lara_tcnn', 'lara_cnn', 'tcnn']:
        return TCNN.Lara_TCNN(dim_in, dim_out, 6, window_size)

    # Transformers
    if model_name in ['transformer', 'small_transformer', 'raw_transformer', 'lara_transformer']:
        return Transformer.IMUTransformerEncoder(dim_in, dim_out, window_size, model_name)
    
    # RNNs
    if model_name == 'lstm':
        return LSTM.IMU_LSTM(dim_in, dim_out, window_size, model_name)
    
    raise RuntimeError('{} is not a known model'.format(model_name))