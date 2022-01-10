import copy
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from .dataloaders.lara_loader import load as lara_load
from .dataloaders.motionsense_loader import load as motionsense_load


class IMUDataset(Dataset):
    def __init__(self, features, labels, infos, window_size, window_shift, normalize):
        super(IMUDataset, self).__init__()
        
        window_shift = max(1, min(window_shift, window_size))

        n = labels.shape[0]
        tmp_start_indices = list(range(0, n - window_size, window_shift))

        #Including only Windows that come from the same recording
        self.start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[x+window_size]), tmp_start_indices))
        
        logging.info('N_samples = {}'.format(n))
        logging.info('N_windows = {}'.format(len(self.start_indices)))
        logging.info('Window_size = {}'.format(window_size))
        logging.info('Stepsize = {}'.format(window_shift))

        self.imu = normalize_data(features) if normalize else features
        self.labels = labels
        self.infos = infos
        self.persons = sorted(np.unique(infos[:,0]))
        self.window_size = window_size

    def __len__(self):
        return len(self.start_indices)


    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        win_size = self.window_size
        imu = self.imu[start_idx : start_idx + win_size]
        label = self.labels[start_idx + win_size // 2]
        
        return imu, label


def normalize_data(data):
    for ch in range(data.shape[1]):
        max_ch = np.max(data[:, ch])
        min_ch = np.min(data[:, ch])
        median_old_range = (max_ch + min_ch) / 2
        data[:, ch] = (data[:, ch] - median_old_range) / (max_ch - min_ch)
    return data + 0.01 * torch.randn(data.shape).numpy()


def split_data(dataset, split_ratio=0.1, split_type='person'):
        split_ratio = split_ratio if 0 < split_ratio <= 1 else 0.1
        split_ratio = 1 - split_ratio

        train_data = dataset
        test_data = copy.copy(train_data)

        n = len(train_data)
        split_idx = int(n * split_ratio)
        
        if split_type not in ['person', 'person_random', 'person random']:
            raise RuntimeError('Unknown split_type {} -> see config'.format(split_type))

        if split_type == 'person':
            split_idx = int (len(train_data.persons) * split_ratio)

            train_persons = train_data.persons[:split_idx]
            test_persons = train_data.persons[split_idx:]

            train_data.start_indices = get_indices_for_persons(train_data, train_persons)
            train_data.persons = sorted(train_persons)
            test_data.start_indices = get_indices_for_persons(test_data, test_persons)
            test_data.persons = sorted(test_persons)

        if split_type in ['person_random', 'person random']:
            perm = np.random.RandomState(seed=42).permutation(len(train_data.persons))
            perm = [train_data.persons[i] for i in perm]

            split_idx = int (len(perm) * split_ratio)

            train_persons = perm[:split_idx]
            test_persons = perm[split_idx:]
            
            train_data.start_indices =  get_indices_for_persons(train_data, train_persons)
            train_data.persons = sorted(train_persons)
            test_data.start_indices = get_indices_for_persons(test_data, test_persons)
            test_data.persons = sorted(test_persons)

        return train_data, test_data


def get_indices_for_persons(data, persons):
    return [idx for idx in data.start_indices if data.infos[idx, 0] in persons]


def load_data(dir_path, dataset, classification_type=None):
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

    logging.info('Label_dict = {}'.format(label_dict))

    # Creating the IMU-Dataset
    window_size = data_config.get('window_size')
    if window_size % 2 != 0: 
        logging.warning('Window_size must be even, changing window_size  {} -> {}'.format(window_size, window_size+1))
        window_size += 1
    window_shift = data_config.get('window_shift')
    normalize = data_config.get('normalize')
    imu_dataset = IMUDataset(features, labels, infos, window_size, window_shift, normalize)

    logging.info('Normalize = {}'.format(normalize))

    # Splitting the Data
    test_size = data_config.get('test_size')
    split_type = data_config.get('split_type')
        
    train_data, test_data = split_data(imu_dataset, test_size, split_type)

    return train_data, test_data, label_dict


def preprocessing(features, labels, infos, label_dict):
    logging.info('Preprocessing - Removing Null & None Label')
    label_dict = {k: v.upper() for k, v in label_dict.items()}
    valid_indices = [idx for idx, label in enumerate(labels) if label_dict.get(label) not in ['NULL', 'NONE']]
    features = np.array([features[idx] for idx in valid_indices], dtype=np.float32)
    labels = np.array([labels[idx] for idx in valid_indices], dtype=np.int32)
    infos = np.array([infos[idx] for idx in valid_indices], dtype=np.int32)

    unique_labels = np.unique(labels)
    unique_labels = np.sort(unique_labels)
    n_features = unique_labels.shape[0]
    translation_dict = {idx:unique_labels[idx] for idx in range(n_features)}
    new_dict = {lbl: label_dict[translation_dict[lbl]] for lbl in translation_dict.keys()}

    reverse_translation_dict = {v: k for k, v in translation_dict.items()}
    labels = np.array([reverse_translation_dict[lbl] for lbl in labels], dtype=np.int32)

    return features, labels, infos, new_dict
