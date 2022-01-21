import copy
import json
import argparse
import math
import torch
import numpy as np
import logging
from util import utils
from data.dataloader import retrieve_dataloaders
from os.path import join
import models.model as  model_loader
import sklearn.metrics
import util.functions as functions
import os


def read_config(path):
    with open(os.path.join(path, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    with open(os.path.join(path, 'meta_config.json'), "r") as read_file:
        meta_config = json.load(read_file)
    return config, meta_config 


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


def validation_loop(dataloader, model, device, loss_fn, predict_classes=True):
    # Set to eval mode
    model.eval()
    
    size = len(dataloader.dataset)

    batch_losses = []
    accuracy = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(device).to(dtype=torch.int64) if predict_classes else y.to(device).to(dtype=torch.float32)

            pred = model(X)
            loss = loss_fn(pred, y)
            batch_losses.append(loss.item())

            # Getting the binary coded attribute_vector if needed
            if not predict_classes:
                pred = torch.nn.Sigmoid()(pred)
                # Debugging
                pred = functions.predict_attributes(pred)
                #pred = pred.round()
            else:
                pred = pred.argmax(dim=1)
                
            accuracy += sum([int(torch.equal(x, y)) for x, y in zip(y, pred)])

    batch_losses = np.array(batch_losses)
    return np.mean(batch_losses), accuracy/size


def test_loop(dataloader, model, device, predict_classes=True):
    # Set to eval mode
    model.eval()

    predicted_labels = []
    real_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(dtype=torch.float32)
            y = y.to(dtype=torch.float32)
            pred = model(X)

            # Getting the binary coded attribute_vector if needed
            if not predict_classes:
                pred = torch.nn.Sigmoid()(pred)
                pred = functions.predict_attributes(pred)
                #pred = pred.round()
                pred = pred.cpu().numpy().squeeze()
                real = y.numpy().squeeze()
            else:
                real = y.item()
                pred = pred.argmax(dim=1).item()

            real_labels.append(real)
            predicted_labels.append(pred)

    logging.info('UNIQUE REAL_LABELS SHAPE = {}'.format(np.unique(np.array(real_labels), axis=0).shape))
    logging.info('UNIQUE PREDICTIONS SHAPE = {}'.format(np.unique(np.array(predicted_labels), axis=0).shape))
    
    return np.array([real_labels, predicted_labels], dtype=np.int32)


def run_experiment(config, args):
    dir_path = config['setup']['dir_path']
    
    # Initializing experiment and logger
    run_folder, run_name = utils.init_run(dir_path, args.experiment, args.verbose)
    filename_prefix = join(run_folder, run_name)
    
    # Saving config
    with open(join(filename_prefix + '_config.json'), "w") as f:
        json.dump(config, f, indent=4)
    
    # Initializing Cuda
    device = utils.init_cuda(config['setup']['device_id'], config['setup']['torch_seed'])
    logging.info('Device_id = {}'.format(device))

    # Retreiving datasets
    train_dataloader, validation_dataloader, test_dataloader = retrieve_dataloaders(dir_path, config['data'], config['training']['batch_size'])
    train_persons = train_dataloader.dataset.persons if train_dataloader is not None else []
    validation_persons = validation_dataloader.dataset.persons if validation_dataloader is not None else []
    test_persons = test_dataloader.dataset.persons if test_dataloader is not None else []
    
    logging.info('Train_data_persons = {} | Validation_data_persons = {} | Test_data_persons = {}'.format(train_persons, validation_persons, test_persons))
    
    predict_classes = config['data']['classification_type'] == 'classes'

    # Choosing Model
    model_name = config['data']['model_name']
    dataset = train_dataloader.dataset if train_dataloader is not None else (validation_dataloader if validation_dataloader is not None else test_dataloader.dataset)
    label_dict = dataset.label_dict
    dim_in = dataset.imu.shape[1]
    dim_out = len(label_dict) if predict_classes else dataset.labels.shape[1]
    window_size = dataset.window_size
    
    model = model_loader.get_model(model_name, dim_in, dim_out, window_size).to(device)
            
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model: {} with {} parameters.'.format(model, pytorch_total_params))

    if train_dataloader is not None:
        # Preparing training
        train_cfg = config['training']

        # Set the loss
        if train_cfg['use_weights_on_loss']:
            if predict_classes:
                weight_vec = functions.get_weights(train_dataloader, validation_dataloader, 'weights').to(device)
                logging.info('Weights = {}'.format([x.item() for x in weight_vec]))
                loss_fn = torch.nn.CrossEntropyLoss(weight=weight_vec)
            else:
                pos_weight_vec = functions.get_weights(train_dataloader, validation_dataloader, 'pos_weights').to(device)
                logging.info('Pos_Weights = {}'.format([x.item() for x in pos_weight_vec]))
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec)
        else:
            loss_fn = torch.nn.CrossEntropyLoss() if predict_classes else torch.nn.BCEWithLogitsLoss()
        logging.info('Loss = {}'.format(loss_fn))

        # Set the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], eps= train_cfg['eps'], weight_decay=train_cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_scheduler_step_size'], gamma=train_cfg['lr_scheduler_gamma'])
        
        logging.info("Training ...")

        # Tracking Loss and best Model
        train_loss_avg_prog = []
        best_model_state = copy.deepcopy(model.state_dict())        
        best_epoch = 0
        if validation_dataloader is not None:
            val_loss_avg_prog = []
            val_acc_prog = []
            best_model_loss = math.inf

        # Training
        evaluation_str = 'Epoch[{:02d}]: Loss = {:.3f} | Val_Loss = {:.3f} | Val_Acc = {:.3f}'
        for epoch in range(max(0, train_cfg['n_epochs'])):
            loss = train_loop(train_dataloader, model, device, loss_fn, optimizer)
            
            if validation_dataloader is None:
                logging.info('Epoch[{:02d}]: Loss = {:.3f}'.format(epoch, loss))
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                
            else:
                val_loss, val_acc = validation_loop(validation_dataloader, model, device, loss_fn, predict_classes)
                logging.info(evaluation_str.format(epoch, loss, val_loss, val_acc))

                # Tracking some stats
                train_loss_avg_prog.append(loss)
                val_loss_avg_prog.append(val_loss)
                val_acc_prog.append(val_acc)
                        
                # Update best model yet
                if val_loss < best_model_loss:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_model_loss = val_loss
                    best_epoch = epoch
            
            # Update scheduler
            scheduler.step()

        logging.info('Most successful epoch = {:2d}'.format(best_epoch))
        model.load_state_dict(best_model_state)
        logging.info('Training done!')

        stats = np.array([train_loss_avg_prog, val_loss_avg_prog, val_acc_prog], dtype=np.float32) if validation_dataloader is not None else np.array([train_loss_avg_prog], dtype=np.float32)
        np.save(filename_prefix + '_loss.npy', stats)
        torch.save(model.state_dict(), join(dir_path, filename_prefix + '_model.pth'))

    if test_dataloader is not None:
        logging.info("Testing ...")
        classifications = test_loop(test_dataloader, model, device, predict_classes)
        logging.info("Testing complete!")

        np.save(filename_prefix + '_classifications.npy', classifications)
        
        if predict_classes:
            confusion_matr = sklearn.metrics.confusion_matrix(classifications[0], classifications[1])
            logging.info('Confusion_matrix:\n {}'.format(confusion_matr))
            
            l = [label_dict[idx] for idx in range(len(label_dict))]
            classification_report = sklearn.metrics.classification_report(classifications[0], classifications[1], target_names=l, zero_division=0)
            logging.info('Classification_report:\n{}'.format(classification_report))
        
        acc = functions.accuracy_(classifications)
        logging.info('Accuracy = {:.3f}'.format(acc))
    logging.info('Run finished!')


def main():
    root_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Creating configs if needed
    if (utils.init_configs(root_path)) < 0 :
        raise Warning('Files config.json and/or meta_config.json have/has been generated. Please set them up as needed and run again!')
    
    # Reading config.json
    config, meta_config = read_config(root_path)
    
    # Adding arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--run_meta', action='store_true', help='Choose if you want to run the meta_config')
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='Specify if logging outputs should also be printed in stdout')
    arg_parser.add_argument('--experiment', default=None, help='Name for the experiment')
    
    # Parsing arguments
    args = arg_parser.parse_args()
    
    # Initializing Directory
    dir_path = config['setup']['dir_path']
    utils.init_dir_structure(dir_path, args.experiment)
    
    # Starting the run(s)
    if args.run_meta: 
        # Iterating over all Settings:
        for model_name in meta_config['model_name']:
            for normalize in meta_config['normalize']:
                for window_size in meta_config['window_size']:
                    for split_type in meta_config['split_type']:
                        for torch_seed in meta_config['torch_seed']:
                            run_cfg = copy.deepcopy(config)
                                    
                            run_cfg['data']['model_name'] = model_name
                            run_cfg['data']['normalize'] = normalize
                            run_cfg['data']['window_size'] = window_size
                            run_cfg['data']['split_type'] = split_type
                            run_cfg['setup']['torch_seed'] = torch_seed
                            
                            run_experiment(run_cfg, args)
    else:
        # Running only the specified run of config.json
        run_experiment(config, args)
    

if __name__ == '__main__':
    main()