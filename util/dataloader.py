import copy
import logging
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataset import T
from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load
from .dataloaders.orderpicking_loader import load as orderpicking_load

class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, features, labels, infos, window_size, window_shift, labeling_mode, normalize):
        super(IMUDataset, self).__init__()
        
        if window_shift > 0:
            if window_shift < 1:
                window_shift = max(1, int(window_size * window_shift))
            else:
                window_shift = min(window_shift, window_size)
        else:
            raise 'window_shift {} isnt valid'.format(window_shift)

        if normalize:
            self.imu = normalize_data(self.imu)

        n = labels.shape[0]
        tmp_start_indices = list(range(0, n - window_size + 1, window_shift))

        #Including only Windows that come from the same recording
        self.start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[min(x+window_size, infos.shape[0]-1)]),tmp_start_indices))

        logging.info('n_samples = {}'.format(n))
        logging.info('n_windows = {}'.format(len(self.start_indices)))
        logging.info('window_size = {}'.format(window_size))
        logging.info('stepsize = {}'.format(window_shift))

        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.flatten()

        self.imu = features
        self.labels = labels
        self.infos = infos
        self.labeling_mode = labeling_mode
        self.window_size = window_size

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, (start_index + self.window_size)))
        imu = self.imu[window_indices]
        window_labels = self.labels[window_indices]
        #Choosing label
        
        labeling_mode = self.labeling_mode
        if labeling_mode not in ['first', 'middle', 'last', 'mode']:
            raise RuntimeError('labeling_mode {} not valid'.format(labeling_mode))

        if labeling_mode == 'first':
            label = window_labels[0]
            
        if labeling_mode == 'middle':
            label = window_labels[int(len(window_labels) / 2)]
            
        if labeling_mode ==  'last':
            label = window_labels[-1]

        sample = {'imu': imu,
                  'label': label}
        return sample


def normalize_data(data):
    return data


def split_data(dataset, test_size=0.1, split_type='person', np_seed=42):
        test_ratio = test_size if 0 < test_size <= 1 else 0.1
        train_ratio = 1 - test_ratio

        train_data = dataset
        test_data = copy.copy(train_data)

        n = len(train_data)
        split_idx = int(n * train_ratio)
        
        if split_type not in ['random', 'person']:
            raise RuntimeError('Unknown split_type {} -> see config'.format(split_type))

        if split_type == 'random':
            start_indices_permutation = np.random.RandomState(seed=np_seed).permutation(n)
            start_indices = [train_data.start_indices[i] for i in start_indices_permutation]

            train_data.start_indices = start_indices[:split_idx] 
            test_data.start_indices = start_indices[split_idx:]

        if split_type == 'person':
            persons_unique = np.unique(train_data.infos[:,0])
            split_idx = int (len(persons_unique) * train_ratio)

            train_persons = persons_unique[:split_idx]
            test_persons = persons_unique[split_idx:]
            
            train_data.start_indices =  [idx for idx in train_data.start_indices if train_data.infos[idx, 0] in train_persons]
            test_data.start_indices = [idx for idx in test_data.start_indices if test_data.infos[idx, 0] in test_persons]

        return train_data, test_data


def load_data(dir_path, dataset, classification_type):
    if dataset == 'motionsense':  
        return motionsense_load(dir_path)
    if dataset == 'lara': 
        return lara_load(dir_path, classification_type)   
    if dataset == 'orderpicking': 
        return orderpicking_load(dir_path)
    raise 'dataloader {} does not exist'.format(dataset)


def get_data(dir_path, np_seed, data_config, split=True):
    # Loading data from disc
    dataset = data_config.get('dataset')
    classification_type = data_config.get('classification_type')
    features, labels, infos, label_dict = load_data(dir_path, dataset, classification_type)

    # Creating the IMU-Dataset
    window_size = data_config.get('window_size')
    window_shift = data_config.get('window_shift')
    labeling_mode = data_config.get('labeling_mode')
    normalize = data_config.get('normalize')
    imu_dataset = IMUDataset(features, labels, infos, window_size, window_shift, labeling_mode, normalize)

    if split:
        # Splitting the Data
        test_size = data_config.get('test_size')
        split_type = data_config.get('split_type')
        
        train_data, test_data = split_data(imu_dataset, test_size, split_type, np_seed)

        return train_data, test_data, label_dict

    else:
        return imu_dataset, label_dict

