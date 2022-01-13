import logging
import os
from os.path import join
import numpy as np
import pandas as pd

# import util.utils as utility_functions


def load(path_to_data, classification_type='attributes'):
    
    load_attributes = classification_type == 'attributes'
    path = join(path_to_data, 'data', 'lara')

    directories = ['S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
    scenarios = ['L{num:02d}'.format(num=x) for x in [1,2,3]]
    recordings = ['R{num:02d}'.format(num=x) for x in range(1,31)]
    if load_attributes:
        label_dict = None
    else:
        label_dict = {0:'Standing', 1:'Walking', 2:'Cart', 3:'Handling(up)', 4:'Handling(cntr)', 5:'Handling(down)', 6:'Synchronization', 7:'None'} 

    features = np.zeros((0,30), dtype=np.float64)
    labels = np.zeros((0, 19 if load_attributes else 1), dtype=np.int64)
    infos = np.zeros((0, 2), dtype=np.int64)

    if not os.path.exists(path):
        logging.info('lara_data not found under {}'.format(path))
        utility_functions.download_url('https://zenodo.org/record/3862782/files/IMU%20data.zip?download=1', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'IMU data'), join(path_to_data, 'data', 'lara'))
    logging.info('Loading data from {}.'.format(path))

    
    for dir in directories:
        for sc in scenarios:
            for rec in recordings:
                path_features = join(path, dir, sc + '_' + dir + '_' + rec + '.csv')
                path_labels = join(path, dir, sc + '_' + dir + '_' + rec + '_labels' + '.csv')

                if os.path.isfile(path_features) and os.path.isfile(path_labels):
                    raw_vals = pd.read_csv(path_features)
                    raw_vals = raw_vals.drop(columns=['Time'])
                    
                    raw_lbls = pd.read_csv(path_labels)
                    raw_lbls = raw_lbls.drop(columns=['Class']) if load_attributes else raw_lbls[['Class']]

                    assert len(raw_lbls) == len(raw_vals)

                    infs = np.zeros((len(raw_lbls), 2), dtype=np.int64)
                    infs[:,0] = int(dir[1:3])
                    infs[:,1] = int(rec[1:3])

                    features = np.append(features, np.array(raw_vals), axis=0)
                    labels = np.append(labels, np.array(raw_lbls), axis=0)
                    infos = np.append(infos, infs, axis=0)

    
    return features, labels, infos, label_dict