import json
from sklearn.metrics.pairwise import euclidean_distances
import torch
import numpy as np
import logging
from util import utils
import os
from util.dataloader import get_data
from os.path import join
import models.model as  model_loader
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance

def pred_attribute(arr, pred_type, attr_combinations=None):
    if pred_type == 'rounding':
        for idx in range(arr.shape[0]):
            arr[idx] = 0 if arr[idx] < .5 else 1
        return arr
    if pred_type == 'nn':
        dists = np.array([distance.euclidean(arr, y) for y in attr_combinations])
        min_idx = np.argmin(dists)
        return attr_combinations[min_idx]

def run():
    #Reading config.json
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    dir_path = config.get('setup').get('dir_path')
    classification_type = config.get('data').get('classification_type')

    #Initializing Directory
    utils.init_dir_structure(dir_path)

    #Initializing Logger
    utils.init_logger(dir_path)

    #Initializing Cuda
    device, device_id = utils.init_cuda(config.get('setup').get('device_id'))
    
    #Retreiving datasets
    train_data, test_data, label_dict = get_data(dir_path, config.get('setup').get('np_seed'), config.get('data'))

    logging.info('len(train_data) = {}'.format(len(train_data)))
    logging.info('len(test_data) = {}'.format(len(test_data)))

    #Choosing Model
    model_name = config.get('settings').get('model_name')
    in_dim = train_data.imu.shape[1]
    out_size = 1 if train_data.labels.ndim == 1 else train_data.labels.shape[1]
    win_size = train_data.window_size
    n_classes = len(label_dict)
    model_cfg = config.get('model')
    
    model = model_loader.get_model(model_name, in_dim, out_size, win_size, n_classes, model_cfg).to(device)

    #Loading checkpoint if needed (Empty file name -> No Checkpoint will be loaded)
    if config.get('settings').get('load_checkpoint') != '':
        utils.load_checkpoint(model, dir_path, config.get('settings').get('load_checkpoint'), device_id)

    #Training/Evaluating Model
    if config.get('settings').get('mode') == 'train' or config.get('settings').get('mode') == 'train_test':
        training_cfg = config.get('training')
        # Set to train mode
        model.train()

        # Set the loss
        loss = torch.nn.NLLLoss() if model.output_size in [1, model.n_classes] else torch.nn.BCELoss()

        logging.info('loss = {}'.format(str(loss)))

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                  lr=training_cfg.get('lr'),
                                  eps=training_cfg.get('eps'),
                                  weight_decay=training_cfg.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=training_cfg.get('lr_scheduler_step_size'),
                                                    gamma=training_cfg.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        logging.info("Start train data preparation")

        loader_params = {'batch_size': training_cfg.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': training_cfg.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_data, **loader_params)
        logging.info("Data preparation completed")

        # Get training details
        n_freq_checkpoint = config.get('setup').get("n_freq_checkpoint")
        n_epochs = training_cfg.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_dir(dir_path,'checkpoints'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        logging.info("Start training")
        for epoch in range(n_epochs):

            for batch_idx, minibatch in enumerate(dataloader):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                if model.output_size == 1:
                    label = minibatch.get('label').to(device).to(dtype=torch.long)
                else:
                    if model.output_size == model.n_classes: 
                        label = torch.argmax(minibatch.get('label'), axis=1).to(device).to(dtype=torch.long)
                    else:
                        label = minibatch.get('label').to(device).to(dtype=torch.float32)
                batch_size = label.shape[0]

                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                batch_loss = criterion.item()
                loss_vals.append(batch_loss)
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()


                # Record loss on train set
                if batch_idx % training_cfg.get('n_freq_print') == 0:
                    logging.info("[Batch-{}/Epoch-{}] batch loss: {:.3f}".format(
                                                                        batch_idx+1, epoch+1,
                                                                        batch_loss))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

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

        metric = []
        logging.info("Start testing")
        accuracy_per_label = np.zeros(n_classes)
        count_per_label = np.zeros(n_classes)
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long) if  model.output_size == 1 else minibatch.get('label').to(device).to(dtype=torch.float32)

                # Forward pass
                res = model(minibatch)

                # Evaluate and append
                if classification_type == 'classes':
                    pred_label = res.flatten().cpu().numpy()
                    pred_label = np.argmax(pred_label)
                    real_label = label[0].cpu().numpy()
                    if  config.get('data').get('one_hot_encoding'):
                        real_label = np.argmax(real_label)

                    predicted.append(pred_label)
                    ground_truth.append(real_label)
                    curr_metric = int(pred_label==real_label)
                    metric.append(curr_metric)
                    accuracy_per_label[real_label] += curr_metric
                    count_per_label[real_label] += 1

                else:
                    pred_type = config.get('settings').get('attr_prediction_type')
                    pred_label = res.flatten().cpu().numpy()
                    pred_label = pred_attribute(pred_label, pred_type) if pred_type=='rounding' else pred_attribute(pred_label, pred_type, attr_combinations)
                    real_label = label[0].cpu().numpy()

                    predicted.append(pred_label)
                    ground_truth.append(real_label)
                    curr_metric = int(np.array_equal(pred_label, real_label))
                    metric.append(curr_metric)
        # Record overall statistics
        stats_msg = "Performance of {} on {}".format(model_name, config.get('data').get('dataset'))
        stats_msg = stats_msg + "\n\tAccuracy: {:.3f}".format(np.mean(metric))
        logging.info(stats_msg)

        if classification_type == 'classes':
            confusion_mat = confusion_matrix(ground_truth, predicted, labels = list(range(n_classes)))
            print(confusion_mat.shape)
            accuracies = []
            for i in range(len(accuracy_per_label)):
                    print("Performance for class [{}] - accuracy {:.3f}".format(label_dict.get(i), accuracy_per_label[i]/count_per_label[i]))
                    accuracies.append(accuracy_per_label[i]/count_per_label[i])
            # save dump
            utils.create_dir(dir_path, 'test_results')
            np.savez(join(dir_path, 'test_results/') + "_test_results_dump", confusion_mat = confusion_mat, accuracies = accuracies, count_per_label=count_per_label, total_acc = np.mean(metric))
        

            # Logging Results
            logging.info('Count_per_label = {}'.format(count_per_label))
            logging.info('Confusion-Matrix:\n{}'.format(confusion_mat))
    

if __name__ == '__main__':
    run()
