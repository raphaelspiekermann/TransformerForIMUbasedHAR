import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, remove
import torch
import urllib.request
from tqdm import tqdm
import zipfile


def create_dir(path, name):
    out_dir = join(path, name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger(dir_path):
    path = split(realpath(__file__))[0]
    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        run_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path = create_dir(join(dir_path, 'runs'), run_name)

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(path, '{}.log'.format(run_name))
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)
        return run_name


def init_dir_structure(path):
    parent_dir = split(path)[0]
    dir_name = split(path)[1]
    create_dir(parent_dir, dir_name)
    data_dir = join(parent_dir, dir_name)
    create_dir(data_dir, 'runs')
    create_dir(data_dir, 'data')


def init_cuda(device_id_cfg, torch_seed):
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch.random.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = device_id_cfg
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
