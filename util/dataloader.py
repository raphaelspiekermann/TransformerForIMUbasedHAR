from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load
from .utils import create_output_dir

def load_data(path_to_data, loader_name='lara', classification_type='attributes'):
    create_output_dir(path_to_data, 'input')

    if loader_name == 'motionsense':
        motionsense_load(path_to_data)
        return
    if loader_name == 'lara':
        lara_load(path_to_data, classification_type)
        return
    else:
        raise 'dataloader {} does not exist'.format(loader_name)
