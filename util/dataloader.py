import copy
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load
from .dataloaders.orderpicking_loader import load as orderpicking_load

class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, features, labels, infos, window_size, window_shift, normalize):
        super(IMUDataset, self).__init__()
        
        if window_shift > 0:
            if window_shift < 1:
                window_shift = max(1, int(window_size * window_shift))
            else:
                window_shift = min(window_shift, window_size)
        else:
            raise 'Window_shift {} isnt valid'.format(window_shift)

        n = labels.shape[0]
        tmp_start_indices = list(range(0, n - window_size + 1, window_shift))

        #Including only Windows that come from the same recording
        self.start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[min(x+window_size, infos.shape[0]-1)]),tmp_start_indices))

        logging.info('N_samples = {}'.format(n))
        logging.info('N_windows = {}'.format(len(self.start_indices)))
        logging.info('Window_size = {}'.format(window_size))
        logging.info('Stepsize = {}'.format(window_shift))

        if normalize: normalize_data(features)
        self.imu = features
        self.labels = labels
        self.infos = infos
        self.persons = sorted(np.unique(infos[:,0]))
        self.window_size = window_size
            

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, (start_index + self.window_size)))
        imu = self.imu[window_indices]
        window_labels = self.labels[window_indices]
        
        label = window_labels[int(len(window_labels) / 2)]
        
        return imu, label


def normalize_data(data):
    for ch in range(data.shape[1]):
        max_ch = np.max(data[:, ch])
        min_ch = np.min(data[:, ch])
        median_old_range = (max_ch + min_ch) / 2
        data[:, ch] = (data[:, ch] - median_old_range) / (max_ch - min_ch)
    data += 0.01 * torch.randn(data.shape)


def split_data(dataset, split_ratio=0.1, split_type='person'):
        split_ratio = split_ratio if 0 < split_ratio <= 1 else 0.1
        split_ratio = 1 - split_ratio

        train_data = dataset
        test_data = copy.copy(train_data)

        n = len(train_data)
        split_idx = int(n * split_ratio)
        
        if split_type not in ['person', 'person_random']:
            raise RuntimeError('Unknown split_type {} -> see config'.format(split_type))

        if split_type == 'person':
            split_idx = int (len(train_data.persons) * split_ratio)

            train_persons = train_data.persons[:split_idx]
            test_persons = train_data.persons[split_idx:]

            train_data.start_indices = [idx for idx in train_data.start_indices if train_data.infos[idx, 0] in train_persons]
            train_data.persons = sorted(train_persons)
            test_data.start_indices = [idx for idx in test_data.start_indices if test_data.infos[idx, 0] in test_persons]
            test_data.persons = sorted(test_persons)

        if split_type == 'person_random':
            perm = np.random.RandomState(seed=42).permutation(len(train_data.persons))
            perm = [train_data.persons[i] for i in perm]

            split_idx = int (len(perm) * split_ratio)

            train_persons = perm[:split_idx]
            test_persons = perm[split_idx:]
            
            train_data.start_indices =  [idx for idx in train_data.start_indices if train_data.infos[idx, 0] in train_persons]
            train_data.persons = sorted(train_persons)
            test_data.start_indices = [idx for idx in test_data.start_indices if test_data.infos[idx, 0] in test_persons]
            test_data.persons = sorted(test_persons)

        return train_data, test_data


def load_data(dir_path, dataset, classification_type):
    if dataset == 'motionsense':  
        return motionsense_load(dir_path)
    if dataset == 'lara': 
        return lara_load(dir_path, classification_type)   
    if dataset == 'orderpicking': 
        return orderpicking_load(dir_path)
    raise 'dataloader {} does not exist'.format(dataset)


def get_data(dir_path, data_config=None):
    # Loading data from disc
    dataset = data_config.get('dataset')
    classification_type = data_config.get('classification_type')
    features, labels, infos, label_dict = load_data(dir_path, dataset, classification_type)

    logging.info('Classification_type = {}'.format(classification_type))

    # Preprocessing
    if classification_type == 'classes':
        labels = labels.flatten()
        features, labels, infos, label_dict = preprocessing(features, labels, infos, label_dict)
        labels = one_hot_encoding(labels, len(label_dict))

    logging.info('Label_dict = {}'.format(label_dict))

    # Creating the IMU-Dataset
    window_size = data_config.get('window_size')
    window_shift = data_config.get('window_shift')
    normalize = data_config.get('normalize')
    imu_dataset = IMUDataset(features, labels, infos, window_size, window_shift, normalize)

    if data_config.get('split_type') in ['person', 'person_random']:
        # Splitting the Data
        test_size = data_config.get('test_size')
        split_type = data_config.get('split_type')
        
        train_data, test_data = split_data(imu_dataset, test_size, split_type)

        return train_data, test_data, label_dict

    else:
        return imu_dataset, label_dict


def preprocessing(features, labels, infos, label_dict):
    logging.info('Preprocessing - Removing Null & None Label')
    label_dict = {k: v.upper() for k, v in label_dict.items()}
    valid_indices = [idx for idx, label in enumerate(labels) if label_dict.get(label) not in ['NULL', 'NONE']]
    features = np.array([features[idx] for idx in valid_indices])
    labels = np.array([labels[idx] for idx in valid_indices])
    infos = np.array([infos[idx] for idx in valid_indices])

    unique_labels = np.unique(labels)
    unique_labels = np.sort(unique_labels)
    n_features = unique_labels.shape[0]
    translation_dict = {idx:unique_labels[idx] for idx in range(n_features)}
    new_dict = {lbl: label_dict[translation_dict[lbl]] for lbl in translation_dict.keys()}

    reverse_translation_dict = {v: k for k, v in translation_dict.items()}
    labels = np.array([reverse_translation_dict[lbl] for lbl in labels])

    return features, labels, infos, new_dict


def one_hot_encoding(labels, n):
    encoded_labels = []
    for label in labels:
        lbl = np.zeros(shape=n, dtype=np.int64)
        lbl[label] = 1
        encoded_labels.append(lbl)
    return np.array(encoded_labels, dtype=np.int64)
