import copy
import json
import torch
import pandas as pd
import numpy as np
import logging
from util import utils
import os
from util.dataloader import get_data, split_data
from os.path import join, isfile
import models.model as  model_loader
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch.multiprocessing


def pred_attribute(arr, pred_type, attr_combinations=None):
    if pred_type == 'rounding':
        for idx in range(arr.shape[0]):
            arr[idx] = 0 if arr[idx] < .5 else 1
        return arr
    if pred_type == 'nn':
        dists = np.array([distance.euclidean(arr, y) for y in attr_combinations])
        min_idx = np.argmin(dists)
        return attr_combinations[min_idx]


def eval_run(run_name, dir_path):
    logging.disable(logging.CRITICAL)
    dir_run = join(dir_path, 'runs', run_name)
    model_dict, optim_dict, scheduler_dict, epoch, loss, config = utils.load_checkpoint(dir_path, run_name + '_model')

    # Modify data_config for loading whole dataset
    data_cfg = config['data']
    data_cfg['split_type'] = ''

    # Load dataset
    dataset, label_dict = get_data(dir_path, data_config=config['data'])
    labels = dataset.labels

    # Stats about the dataset

    # Stats about the setup

    # Informations about the run

    # Loss progress
    loss_file = join(dir_run, '{}_loss.npy'.format(run_name))
    if isfile(loss_file):
        plt.figure()

        data = np.load(loss_file)
        loss_prog = data[0,:]
        std_prog = data[1,:]
        val_loss_prog = data[2,:]
        val_std_prog = data[3,:]
        val_acc_prog = data[4,:]
        
        x_vals = np.array(range(data.shape[1]))
        
        #plt.plot(x_vals_1, loss_prog.flatten(), label = 'loss')
        plt.plot(x_vals, loss_prog, label = 'avg_batch_loss')
        #plt.plot(x_vals, std_prog, label = 'std_batch_loss')
        plt.plot(x_vals, val_loss_prog, label = 'avg_batch_loss on validationset')
        #plt.plot(x_vals, val_std_prog, label = 'std_batch_loss on validationset')
        plt.plot(x_vals, val_acc_prog, label = 'accuracy on validationset')

        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.title('Loss_Progress')

        plt.savefig(join(dir_run, 'loss.pdf'))

    # Evaluating the classifications
    classifications_file = join(dir_run, '{}_classifications.npy'.format(run_name))
    if isfile(classifications_file):
        classifications = np.load(classifications_file)
        if config.get('data').get('classification_type') == 'classes':
            classes = np.unique(classifications[0,:])
            class_names = [label_dict[c] for c in classes]
            report = classification_report(y_true=classifications[0, :], y_pred=classifications[1, :], labels=classes, target_names=class_names, zero_division=0, digits=2)
            matr = confusion_matrix(y_true=classifications[0, :], y_pred=classifications[1, :], labels=classes)
            acc = sklearn.metrics.accuracy_score(y_true=classifications[0, :], y_pred=classifications[1, :])
            f1 = sklearn.metrics.f1_score(y_true=classifications[0, :], y_pred=classifications[1, :], average='weighted')
            utils.create_heatmap(classifications[0,:], classifications[1,:], classes, label_dict, 'Acc = {:.3f}   |   w_F1 = {:.3f}'.format(acc, f1), join(dir_run, 'classification_heatmap.pdf'), True)
    
    logging.disable(logging.DEBUG)


def train_loop(dataloader, model, device, loss_fn, optimizer):
    # Set to train mode
    model.train()

    batch_losses = []
            
    for X, y in dataloader:
        X = X.to(device).to(dtype=torch.float32)
        y = y.to(device).to(dtype=torch.float32)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect for recoding and plotting
        batch_losses.append(loss.item())

    batch_losses = np.array(batch_losses)
    return np.mean(batch_losses), np.std(batch_losses)


def validation_loop(dataloader, model, device, loss_fn, predict_attributes=False, attr_pred_fun=lambda x : x):
    # Set to eval mode
    model.eval()
    
    size = len(dataloader.dataset)

    batch_losses = []
    accuracy = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(device).to(dtype=torch.float32)
            pred = model(X)
            batch_losses.append(loss_fn(pred, y).item())

            # Getting the binary coded attribute_vector if needed
            if predict_attributes:
                sig = torch.nn.Sigmoid()
                pred = sig(pred)
                attr_pred_fun(pred)

            else:
                pred = pred.argmax(dim=1)
                if pred.ndim < y.ndim:
                    y = y.argmax(dim=1)
            
            accuracy += sum([int(torch.equal(x, y)) for x, y in zip(y, pred)])

    batch_losses = np.array(batch_losses)
    return np.mean(batch_losses), np.std(batch_losses), accuracy/size


def test_loop(dataloader, model, device, predict_attributes=False, attr_pred_fun=lambda x : x):
    # Set to eval mode
    model.eval()
    
    if predict_attributes:
        sig = torch.nn.Sigmoid()

    predicted_labels = []
    real_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(device).to(dtype=torch.float32)
            pred = model(X)

            # Getting the binary coded attribute_vector if needed
            if predict_attributes:
                pred = sig(pred)
                attr_pred_fun(pred)
                pred = pred.cpu().numpy()
                real = y 
            else:
                real = y.argmax(dim=1).item() if y.ndim > 1 else y.item()
                pred = pred.argmax(dim=1).item()

            real_labels.append(real)
            predicted_labels.append(pred)
    
    return np.array([real_labels, predicted_labels], dtype=np.int64)


def read_config(meta_cfg=False):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    if meta_cfg:
        with open(os.path.join(__location__, 'meta_config.json'), "r") as read_file:
            meta_config = json.load(read_file)
    return config, meta_config if meta_cfg else config


def get_loss(model):
    return torch.nn.CrossEntropyLoss() if model.output_dim == model.n_classes else torch.nn.BCEWithLogitsLoss()
    

def get_optimizer(model, lr, eps, weight_decay, optimizer='Adam'):
    if optimizer.lower()=='adam':
        return torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    if optimizer.lower()=='sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)


def predict_attribute(pred_type):
    if pred_type == 'rounding':
        return lambda x: x.round_()
        
    if pred_type == 'nn':
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        attribute_combinations = pd.read_csv(join(__location__, 'attr_per_class_dataset.csv'))
        attribute_combinations = np.array(attribute_combinations)
        print(attribute_combinations.shape)
        attribute_combinations = attribute_combinations[:, 1:]
        print(attribute_combinations.shape)
        attribute_combinations = np.unique(attribute_combinations, axis=0)
        print(attribute_combinations.shape)

        def f(t):
            device = t.device
            dists = distance.cdist(t.cpu().numpy(), attribute_combinations, metric='euclidean')
            attr_combinations = torch.tensor(attribute_combinations, device=device)
            dists = torch.tensor(dists, device=device)
            min_indices = torch.argmin(dists, dim=1)
            for idx in range(min_indices.shape[0]):
                t[idx] = attr_combinations[min_indices[idx]]
        return lambda x: f(x)


def main():
    # Reading config.json
    config, meta_config = read_config(meta_cfg=True)
    dir_path = config['setup']['dir_path']

    # Initializing Directory
    utils.init_dir_structure(dir_path)

    # Iterating over all Settings:
    for model_name in meta_config['model_name']:
        for split_type in meta_config['split_type']:
            for normalize in meta_config['normalize']:
                for window_size in meta_config['window_size']:
                    for encode_position in meta_config['encode_position']:
                        for torch_seed in meta_config['torch_seed']:
                            run_cfg = copy.copy(config)

                            run_cfg['settings']['model_name'] = model_name
                            run_cfg['data']['split_type'] = split_type
                            run_cfg['data']['normalize'] = normalize
                            run_cfg['data']['window_size'] = window_size
                            run_cfg['model']['encode_position'] = encode_position
                            run_cfg['setup']['torch_seed'] = torch_seed

                            run(run_cfg)

def run(config):
    dir_path = config['setup']['dir_path']

    # Initializing Logger
    run_name = utils.init_logger(dir_path)
    run_folder = join(dir_path, 'runs', run_name)
    
    # Initializing Cuda
    device, device_id = utils.init_cuda(config['setup']['device_id'], config['setup']['torch_seed'])
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info('Device_id = {}'.format(device_id))

    # Retreiving datasets
    train_data, test_data, label_dict = get_data(dir_path, config['data'])
    learn_data, validation_data = split_data(train_data, config['data']['validation_size'], config['data']['split_type'])
    logging.info('Train_data_persons = {} | Validation_data_persons = {} | Test_data_persons = {}'.format(learn_data.persons, validation_data.persons, test_data.persons))

    # Choosing Model
    model_name_ = config['settings']['model_name']
    in_dim_ = train_data.imu.shape[1]
    out_size_ = train_data.labels.shape[1]
    win_size_ = train_data.window_size
    n_classes_ = len(label_dict) if label_dict != None else -1
    model_cfg_ = config['model']
            
    model = model_loader.get_model(model_name_, in_dim_, out_size_, win_size_, n_classes_, model_cfg_).to(device)
    logging.info('Model = {}'.format(model))
    # Preparing training
    filename_prefix = join(run_folder, run_name)
    train_cfg = config['training']

    # Set the loss
    loss_fn = get_loss(model)

    # Set the optimizer and scheduler
    optimizer = get_optimizer(model, train_cfg['lr'], train_cfg['eps'], train_cfg['weight_decay'], train_cfg['optimizer'])
    scheduler = get_scheduler(optimizer, train_cfg['lr_scheduler_step_size'], train_cfg['lr_scheduler_gamma'])

    # Set the dataset and data loader
    logging.info("Start train data preparation")

    loader_params = {'batch_size': train_cfg['batch_size'], 'shuffle': True, 'num_workers': train_cfg['n_workers']}
    train_dataloader = torch.utils.data.DataLoader(learn_data, **loader_params)
    valid_dataloader = torch.utils.data.DataLoader(validation_data, **loader_params)
    logging.info("Data preparation completed")

    # Tracking Loss 
    train_loss_avg_prog = []
    train_loss_std_prog = []
    val_loss_avg_prog = []
    val_loss_std_prog = []
    val_acc_prog = []
    
    # Best Model
    best_model_state = copy.copy(model.state_dict)
    best_model_loss = 1000
    best_epoch = 0
    
    # Patience for early stopping
    patience = 0

    # Train_loop
    for epoch in range(max(1, train_cfg['n_epochs'])):
        epoch_loss, epoch_std = train_loop(train_dataloader, model, device, loss_fn, optimizer)

        if config['data']['classification_type'] == 'classes':
            avg, std, acc = validation_loop(valid_dataloader, model, device, loss_fn)
        else:
            attr_pred_fun = predict_attribute(config['settings']['attr_prediction_type'])
            avg, std, acc = validation_loop(valid_dataloader, model, device, loss_fn, True, attr_pred_fun)

        logging.info('Epoch[{:02d}]: Epoch_loss_avg = {:.3f} | Epoch_loss_std = {:.3f} | Val_loss_avg = {:.3f} | Val_loss_std = {:.3f} | Val_acc = {:.3f}'.format(epoch, epoch_loss, epoch_std, avg, std, acc))

        # Tracking some stats
        train_loss_avg_prog.append(epoch_loss)
        train_loss_std_prog.append(epoch_std)
        val_loss_avg_prog.append(avg)
        val_loss_std_prog.append(std)
        val_acc_prog.append(acc)
    
        # Update scheduler
        scheduler.step()

        # Update best model yet
        if avg <= best_model_loss:
            best_model_state = copy.copy(model.state_dict())
            best_model_loss = avg
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            # 5 Iterations with decreasing validation_loss -> Stop training
            if patience > 4:
                logging.info('Early stopping after {:2d} epochs'.format(epoch))
                break
    
    logging.info('Most successful epoch = {}'.format(best_epoch))
    model.load_state_dict(best_model_state)

    logging.info('Training done!')

    stats = np.array([train_loss_avg_prog, train_loss_std_prog, val_loss_avg_prog, val_loss_std_prog, val_acc_prog], dtype=np.float64)
    np.save(filename_prefix + '_loss.npy', stats)
    utils.save_checkpoint(model, optimizer, scheduler, epoch+1, loss_fn, config, dir_path, filename_prefix + '_model.pth')

    
    logging.info("Start test data preparation")
    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': config.get('training').get('n_workers')}
    test_dataloader = torch.utils.data.DataLoader(test_data, **loader_params)
    logging.info("Data preparation completed")

    if config['data']['classification_type'] == 'classes':
        classifications = test_loop(test_dataloader, model, device)
    else:
        attr_pred_fun = predict_attribute(config['settings']['attr_prediction_type'])
        classifications = test_loop(test_dataloader, model, device, True, attr_pred_fun)
    
    logging.info("Testing complete!")
    np.save(filename_prefix + '_classifications.npy', classifications)

    # Saving config
    with open(join(filename_prefix + '_config.json'), "w") as f:
        json.dump(config, f, indent=4)

    # Evaluation of this run
    eval_run(run_name, dir_path)
    logging.info('Run complete!')


if __name__ == '__main__':
    main()