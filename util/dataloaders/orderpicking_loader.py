from re import X
import numpy as np
import pandas as pd
from os.path import join, exists, isfile
import os
from ..utils import download_url
import logging
import shutil



def load(path_to_data):
    fnames = ['_DO__004_data_labels_every-frame_100.npz', '_DO__011_data_labels_every-frame_100.npz', '_DO__017_data_labels_every-frame_100.npz']

    label_dict = {0: "NULL", 1: "UNKNOWN", 2: "FLIP", 3: "WALK",
                  4: "SEARCH", 5: "PICK", 6: "SCAN", 7: "INFO",
                  8: "COUNT", 9: "CARRY", 10: "ACK"}

    path = join(path_to_data, 'data', 'orderpicking')

    if not exists(path):
        logging.info('motionsense_data not found under {}'.format(path))
        download_url('https://drive.google.com/u/0/uc?id=171juakdAdoNcA252IMzMp2Fj8gLlwRhn&export=download', output_path=join(path_to_data, 'data'), tmp_path=join(path_to_data, 'data', 'tmp'), extract_archive=True)
        os.rename(join(path_to_data, 'data', 'Order_Picking_Dataset'), join(path_to_data, 'data', 'orderpicking'))

    logging.info('[INFO] -- Loading data from {}.'.format(path))
    
    values = []
    labels = []
    infos = []

    for person_id, fname in enumerate(fnames):
        pth = join(path, fname)
        try:
            tmp = np.load(pth)
            tmp_vals = tmp['arr_0'].copy()
            tmp_labels = tmp['arr_1'].copy()
            for val_win, lab_win in zip(tmp_vals, tmp_labels):
                values.append(val_win[0])
                labels.append(lab_win[0])
                infos.append([person_id, 0])
        except:
            raise RuntimeError ('file_name {} not found!'.format(pth))

    values = np.array(values)
    labels = np.array(labels)
    infos = np.array(infos)

    return values, labels, infos, label_dict
