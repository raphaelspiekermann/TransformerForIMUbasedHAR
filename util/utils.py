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
from torch import optim
from tqdm import tqdm
import zipfile
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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


def init_logger(dir_path):
    """
    Initialize the logger and create a time stamp for the file
    return: Path to folder belonging to this run
    """
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


def init_dir_structure(path : str):
    parent_dir = split(path)[0]
    dir_name = split(path)[1]
    create_dir(parent_dir, dir_name)
    data_dir = join(parent_dir, dir_name)
    create_dir(data_dir, 'runs')
    create_dir(data_dir, 'data')
    

def load_checkpoint(dir_path, file_name):
    if isfile(file_name):
        path_to_checkpoint = file_name
    else:
        if file_name.endswith('.pth'):
            file_name = file_name[:-4]
        folder_name = '_'.join(file_name.split('_')[:-1])
        run_folder = join(dir_path, 'runs', folder_name)
        file_name += '.pth'
        path_to_checkpoint = join(run_folder, file_name)
    if isfile(path_to_checkpoint):
        logging.info('Loading checkpoint from {}'.format(path_to_checkpoint))
        checkpoint = torch.load(path_to_checkpoint)
        
        model_dict = checkpoint.get('model_state_dict')
        optimizer_dict = checkpoint.get('optimizer_state_dict')
        scheduler_dict = checkpoint.get('scheduler_state_dict')

        epoch = checkpoint.get('epoch')
        loss = checkpoint.get('loss')
        config = checkpoint.get('config')

        return model_dict, optimizer_dict, scheduler_dict, epoch, loss, config
    else:
        raise RuntimeError('[INFO] -- {} is no valid path to a checkpoint file'.format(path_to_checkpoint))


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, dir_path, file_name):
    if not file_name.endswith('.pth'):
        file_name += '.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch, 
        'loss': loss,
        'config':config}
    torch.save(checkpoint, join(dir_path, 'checkpoints', file_name))


def init_cuda(device_id_cfg):
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch.random.manual_seed(42)
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




# Plotting utils
##########################
def create_heatmap(real_labels, pred_labels, labels, label_dict=None, title=None, file_name=None , normalize=True):

    if label_dict != None:
        real_labels = [label_dict[label] for label in real_labels]
        pred_labels = [label_dict[label] for label in pred_labels]
        labels = [label_dict[label] for label in labels]
    
    confusion_matr = confusion_matrix(y_true=real_labels, y_pred=pred_labels, labels=labels)
    if normalize:
        confusion_matr = np.array(confusion_matr, dtype=np.float64)
        for row in confusion_matr:
            row *= 100 / np.sum(row)
    
    plt.figure(figsize=(16, 9), dpi=120)
    sns.set_theme()
    own_cmap = sns.color_palette("viridis", as_cmap=True) #sns.color_palette("pastel", as_cmap=True)
    
    ax = sns.heatmap(confusion_matr, annot=True, fmt=".2f" if normalize else "d", cmap=own_cmap, cbar=False,
                     linewidths=.5, xticklabels=labels, yticklabels=labels)
    if normalize:
        for t in ax.texts: t.set_text(t.get_text() + "%")
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    
    plt.xlabel("Predicted Labels",rotation=0, fontsize=22)
    plt.ylabel("Real Labels", rotation=90, fontsize=22)
    
    
    ax.set_title(title, fontsize=22)

    if file_name != None:
        plt.savefig(file_name)

