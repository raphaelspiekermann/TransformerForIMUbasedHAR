import json 
from os.path import join
from .IMUCLSBaseline import IMUCLSBaseline
from .IMUTransformerEncoder import IMUTransformerEncoder

def get_model(config):
    path_meta_stuff = join(config.get('dir_path'), 'input/meta_stuff.json')
    with open(path_meta_stuff, "r") as read_file:
        meta_stuff = json.load(read_file)

    config['num_classes'] = meta_stuff.get('n_classes')
    config['input_dim'] = meta_stuff.get('n_dimension')

    model_name = config.get('model_name')
    
    if model_name=='transformer':
        return IMUTransformerEncoder(config)

    if model_name=='clsbaseline':
        return IMUCLSBaseline(config)
