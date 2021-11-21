import json
import logging

from matplotlib.pyplot import cla
from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load
from .utils import create_output_dir


def load_data(path_to_data, loader_name='lara', normalize=False, classification_type='classes'):
    create_output_dir(path_to_data, 'input')
    path_to_meta_stuff = path_to_data + 'input/meta_stuff.json'

    with open(path_to_meta_stuff, "r") as read_file:
        meta_stuff = json.load(read_file)

    current_dataset = meta_stuff.get('dataset')
    current_normalize = meta_stuff.get('normalized')
    current_classification_type = meta_stuff.get('classification_type')

    if loader_name not in ['motionsense', 'lara']:
        raise 'dataloader {} does not exist'.format(loader_name)

    if current_dataset==loader_name and current_normalize==normalize and current_classification_type==classification_type:
        logging.info('Data already loaded - nothing to do here')
    
    else:
        if loader_name == 'motionsense':
            motionsense_load(path_to_data)
            dict_ = {'dataset':'motionsense', 'normalized':normalize, 'n_classes':6, 'n_dimension':6, 'classification_type':'classes'}
            
        if loader_name == 'lara':
            lara_load(path_to_data, classification_type)
            dict_ = {'dataset':'lara', 'normalized':normalize, 'n_classes':8, 'n_dimension':30, 'classification_type':classification_type}

        if normalize:
            normalize_data()

        with open(path_to_meta_stuff, 'w') as f:
            json.dump(dict_, f)


def normalize_data():
    pass 

