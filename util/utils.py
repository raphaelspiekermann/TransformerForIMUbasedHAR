import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath, isfile, isdir
import time
from os import mkdir, remove
import torch
import urllib.request
import zipfile


def create_dir(path, name):
    out_dir = join(path, name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_run(dir_path, experiment, verbose):
    path = split(realpath(__file__))[0]
    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        run_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path = create_dir(join(dir_path, 'runs'), run_name) if experiment is None else create_dir(join(dir_path, 'experiments', experiment), run_name)

        log_config_dict['handlers']['file_handler']['filename'] = join(path, '{}.log'.format(run_name))
        log_config_dict['handlers']['screen_handler']['level'] = 'INFO' if verbose else 'WARNING'
        
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)
        return path, run_name


def init_dir_structure(data_path, experiment=None):
    parent_dir = split(data_path)[0]
    if not isdir(parent_dir):
        raise ValueError('Directory "{}" does not exist -> check config.json.'.format(parent_dir))
    dir_name = split(data_path)[1]
    create_dir(parent_dir, dir_name)
    data_dir = join(parent_dir, dir_name)
    create_dir(data_dir, 'runs')
    create_dir(data_dir, 'data')
    experiment_dir = create_dir(data_dir, 'experiments')
    if experiment is not None:
        create_dir(experiment_dir, experiment)
 

def init_configs(root_path):
    x = 0
    if not isfile(join(root_path, 'config.json')):
        generate_example_config(root_path)
        x -= 1
    if not isfile(join(root_path, 'meta_config.json')):
        generate_example_meta_config(root_path)
        x -= 1
    return x
    

def init_cuda(device_id, torch_seed):
    torch.random.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if device_id == 'cpu':
            logging.warning('Running on CPU, despite CUDA being available!')
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    return device


def download_url(url, output_path, tmp_path=None, extract_archive=False):
    logging.info('Downloading from {}'.format(url))
    filename = tmp_path if extract_archive else output_path
    tmp_path, _ = urllib.request.urlretrieve(url, filename)
    logging.info('File stored at {}'.format(filename))
    if extract_archive:
        logging.info('Extracting {} to {}'.format(tmp_path, output_path))
        with zipfile.ZipFile(tmp_path) as myZip:
            myZip.extractall(output_path)
        remove(tmp_path)


def generate_example_config(path):
    config = {
        "data": {
            "model_name": "Transformer",
            "dataset": "lara",
            "classification_type": "classes",
            "normalize": True,
            "window_size": 100,
            "window_shift": 20,
            "test_size": 0.15,
            "validation_size": 0.25,
            "split_type": "person"
        },
        "training": {
            "use_weights_on_loss": False,
            "batch_size": 128,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "eps": 1e-10,
            "lr_scheduler_step_size": 5,
            "lr_scheduler_gamma": 0.5,
            "n_epochs": 30
        },
        "setup": {
            "dir_path": "ADD PATH HERE",
            "torch_seed": 0,
            "device_id": "cpu"
        }
    }
    
    path = join(path, 'config.json')
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def generate_example_meta_config(path):
    meta_config = {
    "model_name": ["Baseline", "TCNN_1D", "TCNN_2D", "TCNN_2D_no_pooling", "Transformer", "Transformer_ReLU", "Transformer_no_pos_embed", "Transformer_no_input_embed", "LSTM_hidden", "LSTM_token"],
    "normalize": [True],
    "window_size": [100, 74, 50, 24],
    "split_type": ["person"],
    "torch_seed": [0, 42,1337]
    }
    path = join(path, 'meta_config.json')
    with open(path, "w") as f:
        json.dump(meta_config, f, indent=4)
