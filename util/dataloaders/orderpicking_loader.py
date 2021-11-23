import numpy as np
import pandas as pd
from os.path import join, exists, isfile
import os
from ..utils import download_url
import logging
import shutil


def load_data(path_to_data, force_download=False):
    fnames = ['_DO__004_data_labels_every-frame_100.npz', '_DO__011_data_labels_every-frame_100.npz', '_DO__017_data_labels_every-frame_100.npz']

    path = join(path_to_data, 'data', 'orderpicking')

    features = np.zeros((0, 27))
    labels = np.zeros((0, 1))
    infos = np.zeros((0, 1))

    # TODO 
    if False and (force_download or (not exists(path))):
        if not exists(path): logging.info('motionsense_data not found under {}'.format(path))
        if force_download: logging.info('forcing download')
        if exists(path): shutil.rmtree(path)
        download_url('https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'A_DeviceMotion_data'), join(path_to_data, 'data', 'motionsense'))
        if exists(join(path_to_data, 'data', '__MACOSX')): shutil.rmtree(join(path_to_data, 'data', '__MACOSX'))

    logging.info('[INFO] -- Loading data from {}.'.format(path))
    
    for fname in fnames:
        

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
