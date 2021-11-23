import logging
import os
from os.path import join
import numpy as np
import pandas as pd
from ..utils import download_url
import shutil


def load(path_to_data, classification_type='attributes', force_download=False):
    
    load_attributes = classification_type == 'attributes'
    path = join(path_to_data, 'data', 'lara')

    directories = ['S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
    scenarios = ['L{num:02d}'.format(num=x) for x in [1,2,3]]
    recordings = ['R{num:02d}'.format(num=x) for x in range(1,31)]

    label_dict = {'':0, '':1, '':2, '':3, '':4}

    features = np.zeros((0,30))
    labels = np.zeros((0, 19 if load_attributes else 1))
    infos = np.zeros((0, 2))

    if force_download or (not os.path.exists(path)):
        if not os.path.exists(path): logging.info('lara_data not found under {}'.format(path))
        if force_download: logging.info('forcing download')
        if os.path.exists(path): shutil.rmtree(path)
        download_url('https://zenodo.org/record/3862782/files/IMU%20data.zip?download=1', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'IMU Data'), join(path_to_data, 'data', 'lara'))

    logging.info('[INFO] -- Loading data from {}.'.format(path))

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

                    infs = np.zeros((len(raw_lbls), 2))
                    infs[:,0] = int(dir[1:3])
                    infs[:,1] = int(rec[1:3])

                    features = np.append(features, raw_vals.to_numpy(), axis=0)
                    labels = np.append(labels, raw_lbls.to_numpy(), axis=0)
                    infos = np.append(infos, infs, axis=0)

    features = pd.DataFrame(data=features, columns=raw_vals.columns)
    labels = pd.DataFrame(data=labels, columns=raw_lbls.columns)
    infos = pd.DataFrame(data=infos, columns=['person_id', 'recording_nr'])

    path_input = join(path_to_data, 'input')


    # Exporting features, labels and infos as CSVs
    logging.info('Writing features.csv')
    features.to_csv(join(path_input, 'features.csv'), index=False, header=True)

    logging.info('Writing labels.csv')
    labels.to_csv(join(path_input, 'labels.csv'), index=False, header=True)

    logging.info('Writing infos.csv')
    infos.to_csv(join(path_input, 'infos.csv'), index=False, header=True)

    return label_dict
