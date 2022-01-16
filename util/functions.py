import torch
import pandas as pd
import numpy as np
import os
from os.path import join, split
from scipy.spatial import distance


def get_combinations():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    attribute_combinations = pd.read_csv(join(split(__location__)[0], 'data', 'attr_per_class_dataset.csv'), header=None)
    attribute_combinations = np.array(attribute_combinations)
    return np.unique(attribute_combinations[:, 1:], axis=0)


def get_weights(train_dataloader, validation_dataloader, type):
    train_data = train_dataloader.dataset
    validation_data = validation_dataloader.dataset if validation_dataloader is not None else None
    vector_size = np.unique(train_data.labels, axis=0).shape[0] if type == 'weights' else train_data.labels[0].shape[0]
    
    N = len(train_data) + len(validation_data) if validation_dataloader is not None else len(train_data)
    
    if type == 'weights':
        count = [0] * vector_size
        for _,y in train_data:
            count[y] += 1
        if validation_dataloader is not None:
            for _,y in validation_data:
                count[y] += 1
        t = torch.tensor(count)
        weight_vec = N / t
        norm = torch.linalg.norm(weight_vec)
        weight_vec *= vector_size / norm
        return weight_vec
    
    if type == 'pos_weights':
        count = torch.zeros(vector_size)
        for _,y in train_data:
            count += y
        if validation_dataloader is not None:
            for _,y in validation_data:
                count += y
        weight_vec = [(N - x.item()) / x.item() for x in count]
        return torch.tensor(weight_vec)
    
    raise ValueError('Type {} isnt valid'.format(type))


def predict_attributes(t):
    combinations = get_combinations()
    dists = distance.cdist(t.cpu().numpy(), combinations)
    dists = torch.from_numpy(dists).to(t.device)
    min_indices = torch.argmin(dists, dim=1)
    t_new = torch.empty_like(t)
    for idx in range(min_indices.shape[0]):
        min_combination = combinations[min_indices[idx]]
        t_new[idx] = torch.from_numpy(min_combination)
    return t_new.to(device=t.device)

 
def accuracy_(classfcn):
    count = classfcn.shape[1]
    acc = 0
    for real, pred in zip(classfcn[0], classfcn[1]):
        tmp = np.array_equal(real, pred)
        acc += int(tmp)
    return acc / count

