from genericpath import isfile
import numpy as np
import pandas as pd
from os.path import join, exists, isfile
import os
from ..utils import download_url
import logging
import shutil

def load(path_to_data, force_download=False):
    lbl_dirs = ["dws","ups", "wlk", "jog", "std", "sit"]
    lbl_dict = {'dws':0, 'ups':1, 'wlk':2, 'jog':3, 'std':4, 'sit':5}
    recordings = range(1,17)
    subjects = range(1,25)

    path = join(path_to_data, 'data', 'motionsense')

    features = np.zeros((0, 6))
    labels = np.zeros((0, 1))
    infos = np.zeros((0, 2))

    if force_download or (not exists(path)):
        if not exists(path): logging.info('motionsense_data not found under {}'.format(path))
        if force_download: logging.info('forcing download')
        if exists(path): shutil.rmtree(path)
        download_url('https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'A_DeviceMotion_data'), join(path_to_data, 'data', 'motionsense'))
        if exists(join(path_to_data, 'data', '__MACOSX')): shutil.rmtree(join(path_to_data, 'data', '__MACOSX'))

    logging.info('[INFO] -- Loading data from {}.'.format(path))

    for dir in lbl_dirs:
        for rec in recordings:
            for sub in subjects:
                rec_ = '_{}'.format(rec)
                file_name = 'sub_{}.csv'.format(sub)
                path_to_csv = join(path, dir + rec_, file_name)
                if isfile(path_to_csv):
                    dataset = pd.read_csv(path_to_csv) 
                    raw_features = dataset[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z', 'attitude.roll', 'attitude.pitch', 'attitude.yaw']]
                    raw_features = raw_features.rename(columns={
                        'userAcceleration.x' : 'acc.x', 
                        'userAcceleration.y' : 'acc.y',
                        'userAcceleration.z' : 'acc.z', 
                        'attitude.roll' : 'att.roll',
                        'attitude.pitch' : 'att.pitch',
                        'attitude.yaw' : 'att.yaw',
                        })

                    lbls = np.zeros((len(raw_features), 1))
                    lbls[:,0] = lbl_dict[dir]

                    infs = np.zeros((len(raw_features), 2))
                    infs[:,0] = sub
                    infs[:,1] = rec

                    features = np.append(features, raw_features.to_numpy(), axis=0)
                    labels = np.append(labels, lbls, axis=0)
                    infos = np.append(infos, infs, axis=0)

    assert features.shape[0] == labels.shape[0] == infos.shape[0]

    features = pd.DataFrame(data=features, columns=raw_features.columns)
    labels = pd.DataFrame(data=labels, columns=['class'])
    infos = pd.DataFrame(data=infos, columns=['person_id', 'recording_nr'])

    path_input = join(path_to_data, 'input')

    # Exporting features, labels and infos as CSVs
    logging.info('Writing features.csv')
    features.to_csv(join(path_input, 'features.csv'), index=False, header=True)

    logging.info('Writing labels.csv')
    labels.to_csv(join(path_input, 'labels.csv'), index=False, header=True)

    logging.info('Writing infos.csv')
    infos.to_csv(join(path_input, 'infos.csv'), index=False, header=True)

    return lbl_dict
