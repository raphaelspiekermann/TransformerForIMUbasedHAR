import copy
from enum import unique
import logging
from torch.utils.data import Dataset
import numpy as np
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

        n = labels.shape[0]
        tmp_start_indices = list(range(0, n - window_size + 1, window_shift))

        #Including only Windows that come from the same recording
        self.start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[min(x+window_size, infos.shape[0]-1)]),tmp_start_indices))

        logging.info('n_samples = {}'.format(n))
        logging.info('n_windows = {}'.format(len(self.start_indices)))
        logging.info('window_size = {}'.format(window_size))
        logging.info('stepsize = {}'.format(window_shift))
        logging.info('labeling_mode = {}'.format(labeling_mode))

        self.imu = normalize_data(features, normalize)
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


def normalize_data(data, normalize):
    if normalize == '':
        normalize = 'None'
    logging.info('Normalization = {}'.format(normalize))
    if normalize == 'average':
        logging.info('Average-Normalizing of imu-data')
        avg = np.average(data, axis=0)
        data -= avg
        return data

    if normalize == 'min_max':
        logging.info('Min-Max-Normalizing of imu-data')
        for ch in range(data.shape[1]):
            max_ch = np.max(data[:, ch])
            min_ch = np.min(data[:, ch])
            median_old_range = (max_ch + min_ch) / 2
            data[:, ch] = (data[:, ch] - median_old_range) / (max_ch - min_ch)
            return data
    
    return data 


def split_data(dataset, test_size=0.1, split_type='person', np_seed=42):
        test_ratio = test_size if 0 < test_size <= 1 else 0.1
        train_ratio = 1 - test_ratio

        train_data = dataset
        test_data = copy.copy(train_data)

        n = len(train_data)
        split_idx = int(n * train_ratio)
        
        if split_type not in ['random', 'person', 'person_random']:
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
            
            logging.info('train_persons = {}'.format(train_persons))
            logging.info('test_persons = {}'.format(test_persons))

            train_data.start_indices =  [idx for idx in train_data.start_indices if train_data.infos[idx, 0] in train_persons]
            test_data.start_indices = [idx for idx in test_data.start_indices if test_data.infos[idx, 0] in test_persons]

        if split_type == 'person_random':
            persons_unique = np.unique(train_data.infos[:,0])
            perm = np.random.RandomState(seed=np_seed).permutation(len(persons_unique))
            persons_unique = [persons_unique[i] for i in perm]

            split_idx = int (len(persons_unique) * train_ratio)

            train_persons = persons_unique[:split_idx]
            test_persons = persons_unique[split_idx:]

            logging.info('train_persons = {}'.format(train_persons))
            logging.info('test_persons = {}'.format(test_persons))
            
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


def get_data(dir_path, np_seed=42, data_config=None):
    # Loading data from disc
    dataset = data_config.get('dataset')
    classification_type = data_config.get('classification_type')
    features, labels, infos, label_dict = load_data(dir_path, dataset, classification_type)

    #logging.info('label_dict = {}'.format(label_dict))
    #logging.info('N_features = {}'.format(features.shape[0]))
    logging.info('classification_type = {}'.format(classification_type))

    # Preprocessing
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.flatten()

    if classification_type == 'classes':
        features, labels, infos, label_dict = preprocessing(features, labels, infos, label_dict)
    
    logging.info('label_dict = {}'.format(label_dict))
    logging.info('One_hot_encoding = {}'.format(data_config.get('one_hot_encoding')))

    if data_config.get('one_hot_encoding'):
        labels = one_hot_encoding(labels, len(label_dict))


    # Creating the IMU-Dataset
    window_size = data_config.get('window_size')
    window_shift = data_config.get('window_shift')
    labeling_mode = data_config.get('labeling_mode')
    normalize = data_config.get('normalize')
    imu_dataset = IMUDataset(features, labels, infos, window_size, window_shift, labeling_mode, normalize)

    if data_config.get('split_type') in ['random', 'person', 'person_random']:
        # Splitting the Data
        test_size = data_config.get('test_size')
        split_type = data_config.get('split_type')
        
        train_data, test_data = split_data(imu_dataset, test_size, split_type, np_seed)

        return train_data, test_data, label_dict

    else:
        return imu_dataset, label_dict


def preprocessing(features, labels, infos, label_dict):
    logging.info('Preprocessing the data - Removing Null & None Label')
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
