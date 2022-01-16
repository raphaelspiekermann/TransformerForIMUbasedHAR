import logging
import os
from os.path import join
import numpy as np
import pandas as pd
import util.utils as utility_functions


def load(path_to_data, classification_type='classes'):
    path = join(path_to_data, 'data', 'lara')

    directories = ['S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
    scenarios = ['L{num:02d}'.format(num=x) for x in [1,2,3]]
    recordings = ['R{num:02d}'.format(num=x) for x in range(1,31)]
    
    label_dict = {0:'Standing', 1:'Walking', 2:'Cart', 3:'Handling(up)', 4:'Handling(cntr)', 5:'Handling(down)', 6:'Synchronization', 7:'None'}

    features = np.zeros((0,30), dtype=np.float64)
    labels = np.zeros((0, 20), dtype=np.int64)
    infos = np.zeros((0, 3), dtype=np.int64)

    if not os.path.exists(path):
        logging.info('lara_data not found under {}'.format(path))
        utility_functions.download_url('https://zenodo.org/record/3862782/files/IMU%20data.zip?download=1', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'IMU data'), join(path_to_data, 'data', 'lara'))
    logging.info('Loading data from {}.'.format(path))
    
    for dir in directories:
        for sc in scenarios:
            for rec in recordings:
                c = 0 #Needed to track occurrences of None-Class
                path_features = join(path, dir, sc + '_' + dir + '_' + rec + '.csv')
                path_labels = join(path, dir, sc + '_' + dir + '_' + rec + '_labels' + '.csv')

                if os.path.isfile(path_features) and os.path.isfile(path_labels):
                    # Reading data
                    raw_vals = pd.read_csv(path_features)
                    raw_vals = raw_vals.drop(columns=['Time'])
                    raw_vals = np.array(raw_vals)
                    
                    raw_lbls = pd.read_csv(path_labels)
                    raw_lbls = np.array(raw_lbls)
                    
                    assert len(raw_lbls) == len(raw_vals)
                    
                    person = int(dir[1:3])
                    recording = int(rec[1:3])
                    
                    vals = []
                    lbls = []
                    infs = []
                    for X, y in zip(raw_vals, raw_lbls):
                        # 7 is None-Class
                        if y[0] != 7:
                            vals.append(X)
                            lbls.append(y)
                            infs.append([person, recording, c])
                        else:
                            # Tracking None-Class appearances
                            c += 1
                    
                    features = np.append(features, np.array(vals), axis=0)
                    labels = np.append(labels, np.array(lbls), axis=0)
                    infos = np.append(infos, np.array(infs), axis=0)
    
    features, labels, infos, label_dict = preprocessing(features, labels, infos, label_dict)
    labels = labels[:, 0] if classification_type == 'classes' else labels[:, 1:]
    
    return features, labels, infos, label_dict


def preprocessing(features, labels, infos, label_dict):
    logging.info('Preprocessing LARa-Data - Removing None Values')
    classes = labels[:,0]
    valid_indices = [idx for idx, label in enumerate(classes) if label_dict.get(label) != 'None']
    features = np.array([features[idx] for idx in valid_indices], dtype=np.float32)
    labels = np.array([labels[idx] for idx in valid_indices], dtype=np.int32)
    infos = np.array([infos[idx] for idx in valid_indices], dtype=np.int32)

    unique_classes = np.sort(np.unique(classes))
    n_classes = unique_classes.shape[0]
    translation_dict = {idx:unique_classes[idx] for idx in range(n_classes)}
    new_dict = {lbl: label_dict[translation_dict[lbl]] for lbl in translation_dict.keys()}

    reverse_translation_dict = {v: k for k, v in translation_dict.items()}
    labels[:,0] = np.array([reverse_translation_dict[c] for c in classes], dtype=np.int32)

    return features, labels, infos, new_dict
