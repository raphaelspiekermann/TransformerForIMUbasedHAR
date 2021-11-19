import os, os.path
import numpy as np
import pandas as pd




def load(config):
    
    load_attributes = config.get('classification_type') == 'attributes'
    path = config.get("path_to_data_dir") + 'data/lara/'

    directories = ['S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
    scenarios = ['L{num:02d}'.format(num=x) for x in [1,2,3]]
    recordings = ['R{num:02d}'.format(num=x) for x in range(1,31)]


    features = np.zeros((0,30))
    labels = np.zeros((0, 19 if load_attributes else 1))
    infos = np.zeros((0, 2))

    print('[INFO] -- Loading data from {}.'.format(path))

    for dir in directories:
        for sc in scenarios:
            for rec in recordings:
                path_features = path + dir + '/' + sc + '_' + dir + '_' + rec + '.csv'
                path_labels = path + dir + '/' + sc + '_' + dir + '_' + rec + '_labels' + '.csv'

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
    print(features.head())

    labels = pd.DataFrame(data=labels, columns=raw_lbls.columns)
    print(labels.head())

    infos = pd.DataFrame(data=infos, columns=['person_id', 'recording_nr'])
    print(infos.head())

    path_input = config.get("path_to_data_dir") + 'input/'


    # Exporting features, labels and infos as CSVs
    print('[INFO] -- Writing features.csv')
    features.to_csv(path_input + 'features.csv', index=False, header=True)

    print('[INFO] -- Writing labels.csv')
    labels.to_csv(path_input + 'labels.csv', index=False, header=True)

    print('[INFO] -- Writing infos.csv')
    infos.to_csv(path_input + 'infos.csv', index=False, header=True)