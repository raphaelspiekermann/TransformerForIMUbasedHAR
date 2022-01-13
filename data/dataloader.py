import copy
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import data.dataloaders.lara_loader as lara
import data.dataloaders.motionsense_loader as motionsense

class IMUDataset(Dataset):
    def __init__(self, start_indices, imu_data, labels, window_size, label_dict, persons):
        super(IMUDataset, self).__init__()
        self.start_indices = start_indices
        self.imu = imu_data
        self.labels = labels
        self.window_size = window_size
        self.label_dict = label_dict
        self.persons = persons

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        imu = self.imu[start_idx : start_idx + self.window_size]
        label = self.labels[start_idx + self.window_size // 2]
        
        return imu, label


def load_data(dir_path, dataset, classification_type=None):
    if dataset == 'motionsense':  
        return motionsense.load(dir_path)
    if dataset == 'lara': 
        return lara.load(dir_path, classification_type)   
    raise 'Dataloader {} does not exist'.format(dataset)


def retrieve_dataloaders(path, config, batch_size):
    dataset = config['dataset']
    c_type = config['classification_type']
    
    # Loading the raw data
    imu_data, labels, infos, label_dict = load_data(path, dataset, c_type)
    
    # Preprocessing
    if c_type == 'classes':
        labels = labels.flatten()
        imu_data, labels, infos, label_dict = preprocessing(imu_data, labels, infos, label_dict)
        
    # Normalize data
    if config['normalize']:
        imu_data = min_max_normalization(imu_data)
        
    # Preperation for splitting the data by recorded persons
    test_size = config['test_size']
    validation_size = config['validation_size']
    assert 0 <= test_size <= 1 and 0 <= validation_size <= 1
    
    # split_type \in {'person', 'person_random'}
    split_type = config['split_type']
    persons = np.unique(infos[:, 0])
    
    if split_type == 'person_random':
        np.random.RandomState(seed=42).shuffle(persons)
    
    # Data -> Train_data | Test_data
    split_idx = int(persons.shape[0] * (1 - test_size))
    test_persons = persons[split_idx:]
    train_persons = persons[:split_idx]
    
    # Train_data -> Train_data | Validation_data
    split_idx = int(train_persons.shape[0] * (1 - validation_size))
    validation_persons = train_persons[split_idx:]
    train_persons = train_persons[:split_idx]
    
    # Sorting
    train_persons = sorted(train_persons)
    validation_persons = sorted(validation_persons)
    test_persons = sorted(test_persons)
    
    # Preperation for sampling the data into fixed sized windows with a arbitrary window_shift/step_size
    window_size = config['window_size']
    window_shift = config['window_shift']
    assert 0 < window_size and 0 < window_shift
    
    # Window_size must be an even integer (see IMU_CLS_Baseline.py)
    if window_size % 2 != 0:
        logging.warning('Window_size must be even, changing window_size  {} -> {}'.format(window_size, window_size+1))
        window_size += 1
        
    # Generating start_indices for each window
    n = labels.shape[0]
    start_indices = list(range(0, n - window_size, window_shift))
    
    #Including only frames comming from the same recording (and same person)
    start_indices = list(filter(lambda x : np.array_equal(infos[x], infos[x+window_size]), start_indices))
    
    # Splitting start_indices for datasets
    start_indices_train_data = [idx for idx in start_indices if infos[idx, 0] in train_persons]
    start_indices_validation_data = [idx for idx in start_indices if infos[idx, 0] in validation_persons]
    start_indices_test_data = [idx for idx in start_indices if infos[idx, 0] in test_persons]
    
    # Logging some usefull informations
    logging.info('Classification_type = {}'.format(c_type))
    logging.info('N_samples = {}'.format(n))
    logging.info('N_windows = {}'.format(len(start_indices)))
    logging.info('Window_size = {}'.format(window_size))
    logging.info('Stepsize = {}'.format(window_shift))
    logging.info('Normalize = {}'.format(config['normalize']))
    
    # Datasets
    train_data = IMUDataset(start_indices_train_data, imu_data, labels, window_size, label_dict, train_persons)
    validation_data = IMUDataset(start_indices_validation_data, imu_data, labels, window_size, label_dict, validation_persons)
    test_data = IMUDataset(start_indices_test_data, imu_data, labels, window_size, label_dict, test_persons)
    
    # Dataloaders
    loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2}
    train_loader = torch.utils.data.DataLoader(train_data, **loader_params)
    validation_loader = torch.utils.data.DataLoader(validation_data, **loader_params) if len(validation_data) > 0 else None
    
    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    test_loader = torch.utils.data.DataLoader(test_data, **loader_params)
    
    # Done
    return train_loader, validation_loader, test_loader


# Preprocessing utility
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


def min_max_normalization(data):
    for ch in range(data.shape[1]):
        max_ch = np.max(data[:, ch])
        min_ch = np.min(data[:, ch])
        data[:, ch] = (data[:, ch] - min_ch) / (max_ch - min_ch)
    return data + 0.01 * torch.randn(data.shape).numpy()