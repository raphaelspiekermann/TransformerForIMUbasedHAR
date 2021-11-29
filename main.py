import json
from numpy.lib.function_base import average
import torch
import numpy as np
import logging
from util import utils
import os
from util.dataloader import get_data, load_data
from os.path import join, isfile
import models.model as  model_loader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sklearn.metrics
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.multiprocessing

def run():
    #Reading config.json
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)

    dir_path = config.get('setup').get('dir_path')
    
    #Initializing Directory
    utils.init_dir_structure(dir_path)

    #Initializing Logger
    run_name = utils.init_logger(dir_path)
    run_folder = join(dir_path, 'runs', run_name)
    
    #Initializing Cuda
    device, device_id = utils.init_cuda(config.get('setup').get('device_id'))
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info('device_id = {}'.format(device_id))

    #Loading checkpoint if needed (Empty file name -> No Checkpoint will be loaded)
    if config.get('settings').get('load_checkpoint') != '':
        model_dict, optim_dict, scheduler_dict, epoch_cp, loss_cp, config_cp =  utils.load_checkpoint(dir_path, config.get('settings').get('load_checkpoint'))
        logging.info('Updating config to fit loaded checkpoint')
        config['data'] = config_cp['data']
        config['settings']['model_name'] = config_cp['settings']['model_name']
        if config.get('settings').get('load_training_settings'):
            logging.info('Updating training settings (except n_epochs)')
            epoch_nr = config.get('training').get('n_epochs')
            config['training'] = config.cp['training']
            config['training']['n_epochs'] = epoch_nr

    #Retreiving datasets
    train_data, test_data, label_dict = get_data(dir_path, config.get('data').get('np_seed'), config.get('data'))
    logging.info('Elements in train_data = {}'.format(len(train_data)))
    logging.info('Elements in test_data = {}'.format(len(test_data)))

    #Choosing Model
    model_name_ = config.get('settings').get('model_name')
    in_dim_ = train_data.imu.shape[1]
    out_size_ = 1 if train_data.labels.ndim == 1 else train_data.labels.shape[1]
    win_size_ = train_data.window_size
    n_classes_ = len(label_dict) if label_dict != None else -1
    model_cfg_ = config.get('model')
            
    model = model_loader.get_model(model_name_, in_dim_, out_size_, win_size_, n_classes_, model_cfg_).to(device)

    if config.get('settings').get('load_checkpoint') != '':
        model.load_state_dict(model_dict)
        
    classification_type = config.get('data').get('classification_type')
    filename_prefix = join(run_folder, run_name)

    #Training/Evaluating Model
    if config.get('settings').get('mode') == 'train' or config.get('settings').get('mode') == 'train_test':
        training_cfg = config.get('training')
        # Set to train mode
        model.train()

        # Set the loss
        if model.output_size == 1:
            loss = torch.nn.NLLLoss()
        else:
            if model.output_size == model.n_classes:
                loss = torch.nn.CrossEntropyLoss()
            else:
                loss = torch.nn.BCELoss()

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                    lr=training_cfg.get('lr'),
                                    eps=training_cfg.get('eps'),
                                    weight_decay=training_cfg.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size=training_cfg.get('lr_scheduler_step_size'),
                                                        gamma=training_cfg.get('lr_scheduler_gamma'))

        if config.get('settings').get('load_checkpoint') != '' and config.get('settings').get('load_training_settings'):
            loss = loss_cp
            optim.load_state_dict(optim_dict)
            scheduler.load_state_dict(scheduler_dict)


        # Set the dataset and data loader
        logging.info("Start train data preparation")

        loader_params = {'batch_size': training_cfg.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': training_cfg.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_data, **loader_params)
        logging.info("Data preparation completed")

        # Get training details
        n_freq_checkpoint = config.get('setup').get("n_freq_checkpoint")
        start_epoch = epoch_cp if config.get('settings').get('load_checkpoint') != '' else 0
        n_epochs = training_cfg.get("n_epochs") + start_epoch
        
        loss_stats = []
        # Train
        logging.info("Start training")
        for epoch in range(start_epoch, n_epochs):
            # Making the sampling deterministic
            torch.random.manual_seed(epoch)
            loss_acc = []
            
            for minibatch in dataloader:
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                if model.output_size == 1:
                    label = minibatch.get('label').to(device).to(dtype=torch.long)
                else:
                    if model.output_size == model.n_classes:
                        label = minibatch.get('label').to(device).to(dtype=torch.float32)
                    else:
                        label = minibatch.get('label').to(device).to(dtype=torch.float32)

                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                loss_acc.append(criterion.item())

                # Back prop
                criterion.backward()
                optim.step()
            
            # Scheduler update
            loss_stats.append(loss_acc)
            loss_acc = np.array(loss_acc)
            average_loss = np.average(loss_acc)
            std_loss = np.std(loss_acc)
            logging.info('[Epoch {:02d}] AVG = {:.3f} | STD = {:.3f}'.format(epoch, average_loss, std_loss))

            scheduler.step()

            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                utils.save_checkpoint(model, optim, scheduler, epoch+1, loss, config, dir_path, filename_prefix + '_checkpoint-{}.pth'.format(epoch))


        logging.info('Training completed')
        logging.info('Saving stats about loss progress')
        loss_stats = np.array(loss_stats, dtype=np.float64)
        np.save(filename_prefix + '_loss.npy', loss_stats)
        utils.save_checkpoint(model, optim, scheduler, epoch+1, loss, config, dir_path, filename_prefix + '_checkpoint-final.pth')

    if config.get('settings').get('mode') == 'test' or config.get('settings').get('mode') == 'train_test':
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        logging.info("Start test data preparation")
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('training').get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_data, **loader_params)
        logging.info("Data preparation completed")

        # computing all attr_combinations
        if model.output_size != 1 and config.get('settings').get('attr_prediction_type') == 'nn': 
            attr_combinations = np.unique(test_data.labels, axis=0)
            attr_combinations = np.append(attr_combinations, np.unique(train_data.labels, axis=0), axis=0)
            attr_combinations = np.unique(attr_combinations, axis=0)

        logging.info("Start testing")
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long) if model.output_size == 1 else minibatch.get('label').to(device).to(dtype=torch.float32)

                # Forward pass
                res = model(minibatch)

                # Evaluate and append
                if classification_type == 'classes':
                    pred_label = res.flatten().cpu().numpy()
                    pred_label = np.argmax(pred_label)
                    real_label = label[0].cpu().numpy()
                    if  config.get('data').get('one_hot_encoding'):
                        real_label = np.argmax(real_label)

                else:
                    pred_type = config.get('settings').get('attr_prediction_type')
                    pred_label = res.flatten().cpu().numpy()
                    pred_label = pred_attribute(pred_label, pred_type) if pred_type=='rounding' else pred_attribute(pred_label, pred_type, attr_combinations)
                    real_label = label[0].cpu().numpy()

                predicted.append(pred_label)
                ground_truth.append(real_label)
        logging.info("Testing complete")
        logging.info('Saving classifications')
        classifications = np.array([ground_truth, predicted], dtype=np.int64)
        np.save(filename_prefix + '_classifications.npy', classifications)

    # saving config
    with open(join(filename_prefix + '_config.json'), "w") as f:
        json.dump(config, f, indent=4)

    # evaluation
    if config.get('settings').get('mode') in ['train', 'train_test']:
        eval_run(run_name, dir_path)


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
    model_dict, optim_dict, scheduler_dict, epoch, loss, config = utils.load_checkpoint(dir_path, run_name + '_checkpoint-final')

    # Modify data_config for loading whole dataset
    data_cfg = config.get('data')
    data_cfg['split_type'] = ''

    # Load dataset
    dataset, label_dict = get_data(dir_path, data_config=config.get('data'))
    labels = dataset.labels

    # Stats about the dataset

    # Stats about the setup

    # Informations about the run

    # Loss progress
    loss_file = join(dir_run, '{}_loss.npy'.format(run_name))
    if isfile(loss_file):
        plt.figure()

        loss_prog = np.load(loss_file) 
        loss_prog_avg = np.average(loss_prog, axis=1)
        loss_prog_std = np.std(loss_prog, axis=1)
        n_epochs, n_batches = (loss_prog.shape[0], loss_prog.shape[1])

        x_vals_1 = np.array(range(n_epochs * n_batches)) / n_batches
        x_vals_2 = np.array(range(1, n_epochs+1)) - .5
        
        plt.plot(x_vals_1, loss_prog.flatten(), label = 'loss')
        plt.plot(x_vals_2, loss_prog_avg, label = 'avg_batch_loss')
        plt.plot(x_vals_2, loss_prog_std, label = 'std_batch_loss')

        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.title('Loss_Function_Progress')

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
            utils.create_heatmap(classifications[0,:], classifications[1,:], classes, label_dict, 'w_Acc = {:.3f}   |   w_F1 = {:.3f}'.format(acc, f1), join(dir_run, 'classification_heatmap.pdf'), True)
    
    logging.disable(logging.DEBUG)

if __name__ == '__main__':
    run()
    #eval_run('28_11_21_16_48_48', 'D:\\transformer_dir')
