'''
Created on Jun 19, 2019

@author: fmoya
'''
import os

import numpy as np
import src.directories as directories


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, ann_type='center'):
        '''
        Constructor
        '''
        self.path = ''
        self.ann_type = ann_type

        def get_labels(self):
            labels_dict = {0: "NULL", 1: "UNKNOWN", 2: "FLIP", 3: "WALK",
                           4: "SEARCH", 5: "PICK", 6: "SCAN", 7: "INFO",
                           8: "COUNT", 9: "CARRY", 10: "ACK"}

            return labels_dict

    def load_data(self, wr='_DO', test_id=3, all_labels=False):
        dictz = {"_DO": {1: "004", 2: "011", 3: "017"}, "_NP": {1: "004", 2: "014", 3: "015"}}
        print("Data: Loading data for dataset: wr {}; test person {}".format(wr, test_id))
        train_ids = list(dictz[wr])
        train_ids.remove(test_id)
        train_list = ["%s__%s_data_labels_every-frame_100.npz" % (wr, dictz[wr][train_ids[i]]) for i in [0, 1]]
        test_list = ["%s__%s_data_labels_every-frame_100.npz" % (wr, dictz[wr][test_id])]

        train_vals = []
        train_labels = []

        print("Data: Loading train data...")

        for path in train_list:
            try:
                tmp = np.load(directories.path_to_resource_folder() + path)
            except FileNotFoundError:
                tmp = np.load(directories.path_to_main_folder() + path)
            vals = tmp["arr_0"].copy()
            labels = tmp["arr_1"].copy()
            tmp.close()


            for i in range(len(labels)):
                train_vals.append(vals[i])

                if all_labels:
                    train_labels.append(labels[i])
                else:
                    if self.ann_type == "center":
                        # It takes the center value as label
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_arg = label_arg[int(label_arg.shape[0] / 2)]
                    elif self.ann_type == "modus":
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_set = None
                    else:
                        raise RuntimeError("unkown annotype")
                    train_labels.append(label_arg)



        # Make train arrays a numpy matrix
        train_vals = np.array(train_vals)
        train_labels = np.array(train_labels)

        # Load the test data
        test_vals = []
        test_labels = []

        print("Data: Loading test-data...")

        try:
            tmp = np.load(directories.path_to_resource_folder() + test_list[0])
        except FileNotFoundError:
            tmp = np.load(directories.path_to_main_folder() + test_list[0])

        vals = tmp["arr_0"].copy()
        labels = tmp["arr_1"].copy()
        tmp.close()

        for i in range(len(labels)):
            test_vals.append(vals[i])
            if all_labels:
                test_labels.append(labels[i])
            else:
                if self.ann_type == "center":
                    # It takes the center value as label
                    label_arg = labels[i].flatten()
                    label_arg = label_arg.astype(int)
                    label_arg = label_arg[int(label_arg.shape[0] / 2)]
                elif self.ann_type == "modus":
                    label_arg = labels[i].flatten()
                    label_arg = label_arg.astype(int)
                    label_set = None
                else:
                    raise RuntimeError("unkown annotype")
                test_labels.append(label_arg)

        # Make train arrays a numpy matrix
        test_vals = np.array(test_vals)
        test_labels = np.array(test_labels)

        print("Data: Test-data done")

        # calculate number of labels
        labels = set([])
        labels = labels.union(set(train_labels.flatten()))
        labels = labels.union(set(test_labels.flatten()))

        # Remove NULL class label -> should be ignored
        labels = sorted(labels)
        if labels[0] == 0:
            labels = labels[1:]

        #
        # Create a class dictionary and save it. It is a mapping from the original labels to the new labels, due that all the
        # labels dont exist in the warehouses
        #
        class_dict = {}
        for i, label in enumerate(labels):
            class_dict[label] = i

        self.class_dict = class_dict

        print("Data: class_dict {}".format(class_dict))

        print("Data: Creating final matrices with new labels and no Null label...")

        train_vals_fl = []
        train_labels_fl = []
        for idx in range(train_labels.shape[0]):
            item = np.copy(train_vals[idx])
            label = train_labels[idx]
            if label == 0:
                continue
            train_vals_fl.append(item)
            train_labels_fl.append(int(class_dict[label]))

        train_vals_fl = np.array(train_vals_fl)
        train_labels_fl = np.array(train_labels_fl)

        test_vals_fl = []
        test_labels_fl = []
        for idx in range(test_labels.shape[0]):
            item = np.copy(test_vals[idx])
            label = test_labels[idx]

            if label == 0:
                continue
            test_vals_fl.append(item)
            test_labels_fl.append(int(class_dict[label]))

        test_vals_fl = np.array(test_vals_fl)
        test_labels_fl = np.array(test_labels_fl)

        print("Data: Done creating final matrices with new labels and no Null label...")

        train_v_b = np.array(train_vals_fl)
        train_l_b = np.array(train_labels_fl)
        test_v_b = np.array(test_vals_fl)
        test_l_b = np.array(test_labels_fl)

        print("Loading data succeeded!")

        return train_v_b, train_l_b, test_v_b, test_l_b, class_dict
