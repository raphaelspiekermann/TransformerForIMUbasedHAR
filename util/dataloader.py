import json
import copy
import logging
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from os.path import join 
import matplotlib.pyplot as plt
from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load

class _IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, dir_path=None, window_size=50, window_shift=.1, labeling_mode='middle', normalize=False, logging_active=True):
        super(_IMUDataset, self).__init__()
        # Initialising variables
        features_path = join(dir_path,'input', 'features.csv')
        labels_path = join(dir_path,'input', 'labels.csv')
        infos_path = join(dir_path,'input', 'infos.csv')

        # Setting window_shift
        if window_shift > 0:
            # Stepsize relative to window_size
            if window_shift < 1:
                window_shift = max(1, int(window_size * window_shift))
            # Stepsize absolute
            else:
                window_shift = min(window_shift, window_size)
        else:
            logging.info('window_shift {} isnt valid'.format(window_shift))
            raise 'window_shift {} isnt valid'.format(window_shift)

        # Reading IMU-data
        if logging_active:
            logging.info('Reading {}'.format(features_path))
        imu = pd.read_csv(features_path)
        self.imu = imu.iloc[:].values
        
        if normalize:
            self.imu = normalize_data(self.imu)
        
        # Reading label-data
        if logging_active:
            logging.info('Reading {}'.format(labels_path))
        labels = pd.read_csv(labels_path)
        self.labels = labels.iloc[:].values

        # Reading informations
        if logging_active:
            logging.info('Reading {}'.format(infos_path))
        infos = pd.read_csv(infos_path)
        infos = infos.to_numpy()

        n = self.labels.shape[0]
        tmp_start_indices = list(range(0, n - window_size + 1, window_shift))

        #Including only Windows that come from the same recording
        self.start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[min(x+window_size, infos.shape[0]-1)]),tmp_start_indices))

        if logging_active:
            logging.info('n_samples = {}'.format(n))
            logging.info('n_windows = {}'.format(len(self.start_indices)))
            logging.info('window_size = {}'.format(window_size))
            logging.info('stepsize = {}'.format(window_shift))

        self.labeling_mode = labeling_mode
        self.window_size = window_size


    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, (start_index + self.window_size)))
        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices, :]
        #Choosing label

        labeling_mode = self.labeling_mode
        if labeling_mode == 'first':
            label = window_labels[0][0]
            
        if labeling_mode == 'middle':
            label = window_labels[int(len(window_labels) / 2)][0]
            
        if labeling_mode ==  'last':
            label = window_labels[-1][0]
        
        #TODO FIX Mode Calc
        if labeling_mode == 'mode':
            from scipy import stats
            label = stats.mode(window_labels)[0]

        sample = {'imu': imu,
                  'label': label}
        return sample

class IMUDataset():
    def __init__(self, config):
        # Init Attributes
        self.dir_path = config.get('dir_path')
        self.dataset = config.get('dataset')
        self.labeling_mode = config.get('labeling_mode')
        self.test_size = config.get('test_size')
        self.classification_type = config.get('classification_type')
        self.window_size = config.get('window_size')
        self.window_shift = config.get('step_size')
        self.np_seed = config.get('np_seed')
        self.force_download = config.get('force_download')
        self.normalize = config.get('normalize')

        #Loading data
        self.label_dict = self.load_data()

        # Generating train- and testdata
        self.train_data, self.test_data = self.split_data()

    def load_data(self):
        path_to_meta_stuff = join(self.dir_path, 'input/meta_stuff.json')

        try:
            with open(path_to_meta_stuff, "r") as read_file:
                meta_stuff = json.load(read_file)
        except:
            with open(path_to_meta_stuff, 'w') as f:
                dict_ = {'dataset':'', 'n_classes':-1, 'n_dimension':-1, 'classification_type':''}
                json.dump(dict_, f)
            with open(path_to_meta_stuff, "r") as read_file:
                meta_stuff = json.load(read_file)

        current_dataset = meta_stuff.get('dataset')
        current_classification_type = meta_stuff.get('classification_type')

        if self.dataset not in ['motionsense', 'lara']:
            raise 'dataloader {} does not exist'.format(self.dataset)

        if (not self.force_download) and current_dataset==self.dataset and current_classification_type==self.classification_type:
            logging.info('Data already loaded - nothing to do here')
        
        else:
            if self.dataset == 'motionsense':
                label_dict = motionsense_load(self.dir_path, self.force_download)
                dict_ = {'dataset':'motionsense', 'n_classes':6, 'n_dimension':6, 'classification_type':'classes'}
                
            if self.dataset == 'lara':
                label_dict = lara_load(self.dir_path, self.classification_type, self.force_download)
                dict_ = {'dataset':'lara', 'n_classes':8, 'n_dimension':30, 'classification_type':self.classification_type}
            
            with open(path_to_meta_stuff, 'w') as f:
                json.dump(dict_, f)
            
            return label_dict


    def split_data(self):
        test_ratio = self.test_size if 0 < self.test_size <= 1 else 0.1
        train_ratio = 1 - test_ratio

        train_data = _IMUDataset(self.dir_path, self.window_size, self.window_shift, self.labeling_mode, self.normalize, logging_active=True)
        test_data = copy.copy(train_data)
        logging.info('label-distribution of original dataset \n{}'.format(self.eval_dataset(train_data, 'orig')))


        n = len(train_data)
        split_idx = int(n * train_ratio)

        np_seed = self.np_seed
        start_indices_permutation = np.random.RandomState(seed=np_seed).permutation(n)
        start_indices = [train_data.start_indices[i] for i in start_indices_permutation]

        logging.info('Creating train_data with ratio = {:.0%}'.format(train_ratio))
        train_data.start_indices = start_indices[:split_idx] 
        logging.info('n_train_data = {}'.format(len(train_data)))
        logging.info('label-distribution of extracted train_data \n{}'.format(self.eval_dataset(train_data, 'train')))

        logging.info('Creating test_data with ratio = {:.0%}'.format(test_ratio))
        test_data.start_indices = start_indices[split_idx:]
        logging.info('n_test_data = {}'.format(len(test_data)))
        logging.info('label-distribution of extracted test_data \n{}'.format(self.eval_dataset(test_data, 'test')))

        
        print("train_data_stuff = {} + {}".format(id(train_data.imu), id(train_data.start_indices)))
        print("test data stuff = {} + {}".format(id(train_data.imu), id(test_data.start_indices)))


        return train_data, test_data

    def eval_dataset(self, dataset, file_name=None):
        labels = [imu['label'] for imu in dataset]

        lbl_set = set(labels)

        lbl_count = {label : 0 for label in lbl_set}
        for label in labels:
            lbl_count[label] += 1

        if file_name is not None:
            vis_data(labels, self.label_dict, join(self.dir_path, 'input', file_name))
        return lbl_count

#TODO
def normalize_data(data):
    return data

def vis_data(labels, label_dict, path):
    x_vals = list(set(labels))
    y_vals = [len(list(filter(lambda y : y==x, labels))) for x in x_vals]   
    print(len(x_vals))
    print(len(y_vals))
    plt.bar(x_vals, y_vals)
    plt.xlabel("X-Werte")
    plt.ylabel("Y-Werte") 
    plt.savefig(path)
