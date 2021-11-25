import json
import torch
import numpy as np
import logging
from util import utils
import os
from util.dataloader import get_data
from os.path import join
import models.model as  model_loader
from sklearn.metrics import confusion_matrix
from sacred import Experiment

#Reading config.json
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, 'config.json'), "r") as read_file:
    config = json.load(read_file)


def run():
    dir_path = config.get('setup').get('dir_path')
    #Initializing Directory
    utils.init_dir_structure(dir_path)

    #Initializing Logger
    utils.init_logger(dir_path)

    #Initializing Cuda
    device, device_id = utils.init_cuda(config.get('setup').get('device_id'))
    
    #Retreiving datasets
    train_data, test_data, label_dict = get_data(dir_path, config.get('setup').get('np_seed'), config.get('data'), split=True)

    print('train_data = {}'.format(len(train_data)))
    print('test_data = {}'.format(len(test_data)))


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
        loss = torch.nn.NLLLoss() if model.output_size == 1 else torch.nn.BCELoss()

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
                label = minibatch.get('label').to(device).to(dtype=torch.long) if  model.output_size == 1 else minibatch.get('label').to(device).to(dtype=torch.float32)
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
                #if batch_idx % n_freq_print == 0:
                if batch_idx == 0:
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
                pred_label = torch.argmax(res)
                predicted.append(pred_label.cpu().numpy())
                ground_truth.append(label[0].item())
                curr_metric = (pred_label==label).to(torch.int)
                label_id = label[0].item()
                accuracy_per_label[label_id] += curr_metric.item()
                count_per_label[label_id] += 1
                metric.append(curr_metric.item())

        # Record overall statistics
        stats_msg = "Performance of {} on {}".format(model_name, config.get('data').get('dataset'))
        confusion_mat = confusion_matrix(ground_truth, predicted, labels = list(range(n_classes)))
        print(confusion_mat.shape)
        stats_msg = stats_msg + "\n\tAccuracy: {:.3f}".format(np.mean(metric))
        accuracies = []
        for i in range(len(accuracy_per_label)):
                print("Performance for class [{}] - accuracy {:.3f}".format(i, accuracy_per_label[i]/count_per_label[i]))
                accuracies.append(accuracy_per_label[i]/count_per_label[i])
        # save dump
        utils.create_dir(dir_path, 'test_results')
        np.savez(join(dir_path, 'test_results/') + "_test_results_dump", confusion_mat = confusion_mat, accuracies = accuracies, count_per_label=count_per_label, total_acc = np.mean(metric))
        logging.info(stats_msg)

        # Logging Results
        logging.info('Count_per_label = {}'.format(count_per_label))
        logging.info('Confusion-Matrix:\n{}'.format(confusion_mat))
    

if __name__ == '__main__':
    run()

