import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath, isfile
import time
from os import mkdir, remove
import torch
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from tqdm import tqdm
import zipfile

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_dir(path, name):
    """
    Create a new directory for outputs, if it does not already exist
    :param path: (str) path to root directory
    :param name: (str) the name of the directory
    :return: the path to the output directory
    """
    out_dir = join(path, name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger(path_to_data):
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]
    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = join(path_to_data, 'logs')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)


def init_dir_structure(path : str):
    parent_dir = split(path)[0]
    dir_name = split(path)[1]
    create_dir(parent_dir, dir_name)
    data_dir = join(parent_dir, dir_name)
    create_dir(data_dir, 'checkpoints')
    create_dir(data_dir, 'data') 
    create_dir(data_dir, 'logs')
    create_dir(data_dir, 'test_results')
    

def load_checkpoint(model, path_to_data, file_name, device_id):
    path_to_checkpoint = file_name if isfile(file_name) else join(path_to_data, 'checkpoints/', file_name)
    if isfile(path_to_checkpoint):
        logging.info('Loading checkpoint from {}'.format(path_to_checkpoint))
        model.load_state_dict(torch.load(path_to_checkpoint, map_location=device_id))
    else:
        raise '[INFO] -- {} is no valid path to a checkpoint file'.format(path_to_checkpoint)


def init_cuda(device_id_cfg):
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = device_id_cfg
    np.random.seed(numpy_seed)
    return torch.device(device_id), device_id


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, tmp_path=None, extract_archive=False):
    logging.info('Downloading from {}'.format(url))
    filename = tmp_path if extract_archive else output_path
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        tmp_path, _ = urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    logging.info('File stored at {}'.format(filename))
    if extract_archive:
        logging.info('Extracting {} to {}'.format(tmp_path, output_path))
        with zipfile.ZipFile(tmp_path) as myZip:
            myZip.extractall(output_path)
        remove(tmp_path)


# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)