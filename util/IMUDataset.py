from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import logging


class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, config, logging_active=True):
        # Initialising variables
        path_to_input_data = config.get('path_to_data') + 'input/'
        features_path = path_to_input_data + 'features.csv'
        labels_path = path_to_input_data + 'labels.csv'
        infos_path = path_to_input_data + 'infos.csv'
        window_size = config.get('window_size')
        window_shift = config.get('step_size')
        labeling_mode = config.get('labeling_mode')

        super(IMUDataset, self).__init__()
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
            label = window_labels[0]
            
        if labeling_mode == 'middle':
            #TODO Remove last brackets
            label = window_labels[int(len(window_labels) / 2)][0]
            
        if labeling_mode ==  'last':
            label = window_labels[-1]
            
        
        if labeling_mode == 'mode':
            from scipy import stats
            label = stats.mode(window_labels)[0]

        sample = {'imu': imu,
                  'label': label}
        return sample