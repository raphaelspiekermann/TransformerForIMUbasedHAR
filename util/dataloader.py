from .lara_loader import load as lara_load
from .motionsense_loader import load as motionsense_load


def load_data(config):
    loader_name = config.get('dataset')
    if loader_name == 'motionsense':
        motionsense_load(config)
        return
    if loader_name == 'lara': 
        lara_load(config)
        return
    else:
        raise 'dataloader {} does not exist'.format(loader_name)
