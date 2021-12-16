import copy
import json
import torch
import pandas as pd
import numpy as np
import logging
from util import utils
import os
from util.dataloader import get_data, split_data
from os.path import join
import models.model as  model_loader
from scipy.spatial import distance


def train_loop(dataloader, model, device, loss_fn, optimizer):
    # Set to train mode
    model.train()

    batch_losses = []

    for X, y in dataloader:
        X = X.to(device).to(dtype=torch.float32)
        y = y.to(device).to(dtype=torch.int64) if str(loss_fn) == 'CrossEntropyLoss()' else y.to(device).to(dtype=torch.float32)

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
    return np.mean(batch_losses)


def validation_loop(dataloader, model, device, loss_fn, attr_pred_fun=None):
    # Set to eval mode
    model.eval()

    if attr_pred_fun is not None:
        sig = torch.nn.Sigmoid()
    
    size = len(dataloader.dataset)

    batch_losses = []
    accuracy = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(device).to(dtype=torch.int64) if str(loss_fn) == 'CrossEntropyLoss()' else y.to(device).to(dtype=torch.float32)

            pred = model(X)
            loss = loss_fn(pred, y)
            batch_losses.append(loss.item())

            # Getting the binary coded attribute_vector if needed
            if attr_pred_fun is not None:
                pred = sig(pred)
                attr_pred_fun(pred)
            else:
                pred = pred.argmax(dim=1)
            
            accuracy += sum([int(torch.equal(x, y)) for x, y in zip(y, pred)])

    batch_losses = np.array(batch_losses)
    return np.mean(batch_losses), accuracy/size


def test_loop(dataloader, model, device, attr_pred_fun=None):
    # Set to eval mode
    model.eval()
    
    if attr_pred_fun is not None:
        sig = torch.nn.Sigmoid()

    predicted_labels = []
    real_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(dtype=torch.float32)
            pred = model(X)

            # Getting the binary coded attribute_vector if needed
            if attr_pred_fun is not None:
                pred = sig(pred)
                attr_pred_fun(pred)
                pred = pred.cpu().numpy()
                real = y.numpy() 
            else:
                real = y.item()
                pred = pred.argmax(dim=1).item()

            real_labels.append(real)
            predicted_labels.append(pred)
    
    return np.array([real_labels, predicted_labels], dtype=np.int32)


def read_config():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    with open(os.path.join(__location__, 'meta_config.json'), "r") as read_file:
        meta_config = json.load(read_file)
    return config, meta_config 


def get_combinations():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    attribute_combinations = pd.read_csv(join(__location__, 'attr_per_class_dataset.csv'))
    attribute_combinations = np.array(attribute_combinations)
    return np.unique(attribute_combinations[:, 1:], axis=0)


def predict_attributes(combinations : np.ndarray):
    def f(t):
        dists = distance.cdist(t.cpu().numpy(), combinations)
        dists = torch.from_numpy(dists).to(t.device)
        min_indices = torch.argmin(dists, dim=1)
        for idx in range(min_indices.shape[0]):
            min_combination = combinations[min_indices[idx]]
            t[idx] = torch.from_numpy(min_combination).to(t.device)
    return f


def main(run_meta=False):
    # Reading config.json
    config, meta_config = read_config()
    dir_path = config['setup']['dir_path']

    # Initializing Directory
    utils.init_dir_structure(dir_path)

    if run_meta:
        # Iterating over all Settings:
        for model_name in meta_config['model_name']:
            for normalize in meta_config['normalize']:
                for window_size in meta_config['window_size']:
                    for split_type in meta_config['split_type']:
                        for torch_seed in meta_config['torch_seed']:
                            run_cfg = copy.copy(config)
                                    
                            run_cfg['data']['model_name'] = model_name
                            run_cfg['data']['normalize'] = normalize
                            run_cfg['data']['window_size'] = window_size
                            run_cfg['data']['split_type'] = split_type
                            run_cfg['setup']['torch_seed'] = torch_seed

                            run(run_cfg)
    else:
        run(config)


def run(config):
    dir_path = config['setup']['dir_path']

    # Initializing Logger
    run_name = utils.init_logger(dir_path)
    run_folder = join(dir_path, 'runs', run_name)
    
    # Initializing Cuda
    device, device_id = utils.init_cuda(config['setup']['device_id'], config['setup']['torch_seed'])
    logging.info('Device_id = {}'.format(device_id))

    # Retreiving datasets
    train_data, test_data, label_dict = get_data(dir_path, config['data'])
    learn_data, validation_data = split_data(train_data, config['data']['validation_size'], config['data']['split_type'])
    logging.info('Train_data_persons = {} | Validation_data_persons = {} | Test_data_persons = {}'.format(learn_data.persons, validation_data.persons, test_data.persons))
    
    predict_classes = config['data']['classification_type'] == 'classes'

    # Choosing Model
    model_name_ = config['data']['model_name']
    in_dim_ = train_data.imu.shape[1]
    out_size_ = len(label_dict) if predict_classes else train_data.labels.shape[1]
    win_size_ = train_data.window_size
            
    model = model_loader.get_model(model_name_, in_dim_, out_size_, win_size_).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model: {} with {} parameters.'.format(model, pytorch_total_params))

    # Preparing training
    filename_prefix = join(run_folder, run_name)
    train_cfg = config['training']

    # Set the loss
    loss_fn = torch.nn.CrossEntropyLoss() if predict_classes else torch.nn.BCEWithLogitsLoss()
    logging.info('Loss = {}'.format(loss_fn))

    # Set the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], eps= train_cfg['eps'], weight_decay=train_cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_scheduler_step_size'], gamma=train_cfg['lr_scheduler_gamma'])

    # Set the dataset and data loader
    loader_params = {'batch_size': train_cfg['batch_size'], 'shuffle': True, 'num_workers': 4}
    train_dataloader = torch.utils.data.DataLoader(learn_data, **loader_params)
    valid_dataloader = torch.utils.data.DataLoader(validation_data, **loader_params)
    
    logging.info("Training ...")

    # Tracking Loss 
    train_loss_avg_prog = []
    val_loss_avg_prog = []
    val_acc_prog = []
    
    # Best Model
    best_model_state = copy.copy(model.state_dict())
    best_model_acc = 0
    best_epoch = 0

    # Patience
    patience = 0
    last_epoch_acc = 0
    
    attr_pred_fun = None if predict_classes else predict_attributes(get_combinations())

    # Training
    evaluation_str = 'Epoch[{:02d}]: Loss = {:.3f} | Val_Loss = {:.3f} | Val_Acc = {:.3f}'
    for epoch in range(max(0, train_cfg['n_epochs'])):
        loss = train_loop(train_dataloader, model, device, loss_fn, optimizer)
        val_loss, val_acc = validation_loop(valid_dataloader, model, device, loss_fn, attr_pred_fun)
        
        logging.info(evaluation_str.format(epoch, loss, val_loss, val_acc))

        # Tracking some stats
        train_loss_avg_prog.append(loss)
        val_loss_avg_prog.append(val_loss)
        val_acc_prog.append(val_acc)
    
        # Update scheduler
        scheduler.step()

        # Update best model yet
        if val_acc >= best_model_acc:
            best_model_state = copy.copy(model.state_dict())
            best_model_acc = val_acc
            best_epoch = epoch
            patience = 0
        else:
            patience = 0 if val_acc >= last_epoch_acc else patience + 1
            if patience >= 3 and epoch >= 10:
                # Stopping after 3 consecutive epochs with descending accuracy
                logging.info('Early stopping after {:2d} epochs'.format(epoch))
                break
        
        last_epoch_acc = val_acc
    
    logging.info('Most successful epoch = {:2d}'.format(best_epoch))
    model.load_state_dict(best_model_state)
    logging.info('Training done!')

    stats = np.array([train_loss_avg_prog, val_loss_avg_prog, val_acc_prog], dtype=np.float32)
    np.save(filename_prefix + '_loss.npy', stats)
    torch.save(model.state_dict(), join(dir_path, filename_prefix + '_model.pth'))

    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    test_dataloader = torch.utils.data.DataLoader(test_data, **loader_params)

    logging.info("Testing ...")
    classifications = test_loop(test_dataloader, model, device, attr_pred_fun)
    logging.info("Testing complete!")

    np.save(filename_prefix + '_classifications.npy', classifications)

    # Saving config
    with open(join(filename_prefix + '_config.json'), "w") as f:
        json.dump(config, f, indent=4)
    
    logging.info('Run finished!')


if __name__ == '__main__':
    main(run_meta=True)