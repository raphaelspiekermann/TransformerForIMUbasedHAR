import json
from matplotlib.pyplot import subplots_adjust
import torch
import numpy as np
import logging
from util import utils
import os
import util.dataloader as dataloader
from os.path import join
from models.IMUTransformerEncoder import IMUTransformerEncoder
from models.IMUCLSBaseline import IMUCLSBaseline
from util.IMUDataset import IMUDataset
from sklearn.metrics import confusion_matrix



def main():
    #Reading config
    config = read_config()

    #Initializing Logger
    utils.init_logger(config.get('path_to_home'), config.get('path_to_data'))

    #Loading data
    if config.get('load_data'):
        path_to_data = config.get('path_to_data')
        loader_name = config.get('dataset')
        classification_type = config.get('classification_type')
        dataloader.load_data(path_to_data, loader_name, classification_type) 
        #Preprocessing
        #TODO
        if config.get('normalize_data'):
            pass

    #Initializing Cuda
    device, device_id = init_cuda(config.get('device_id'))

    #Choosing Model
    if config.get("use_baseline"):
        model = IMUCLSBaseline(config).to(device)
    else:
        model = IMUTransformerEncoder(config).to(device)

    #Loading checkpoint if needed (Empty file name -> No Checkpoint will be loaded)
    if config.get('load_model') != '':
        load_checkpoint(model, config.get('path_to_data'), config.get('load_model'), device_id)
    
    #Loading Dataset
    dataset = split_data(config)

    #Training/Evaluating Model
    if config.get('mode') == 'train':
        train(model, device, dataset, config)
    else:
        if config.get('mode') == 'test':
            test(model, device, dataset, config)
        else:
            if config.get('mode') == 'train_test':
                #TODO full walkthrough
                pass
            else:
                raise 'mode = {} does not exist'.format(config.get('mode'))
                

    #Exporting results

#############
def split_data(config):
    test_size = config.get('test_size')
    test_ratio = test_size if 0 < test_size <= 1 else 0.1
    train_ratio = 1 - test_ratio

    dataset = IMUDataset(config)
    logging.info('labels of orig dataset \n{}'.format(eval_dataset(dataset)))

    n = len(dataset)
    split_idx = int(n * train_ratio)

    np_seed = config.get('np_seed')
    start_indices_permutation = np.random.RandomState(seed=np_seed).permutation(n)
    start_indices = [dataset.start_indices[i] for i in start_indices_permutation]

    if config.get('mode')=='train':
        logging.info('Creating train_data with ratio = {:.0%}'.format(train_ratio))
        dataset.start_indices = start_indices[:split_idx] 
        logging.info('n_train_data = {}'.format(len(dataset)))
        logging.info('labels of extracted train_data \n{}'.format(eval_dataset(dataset)))

    else:
        logging.info('Creating test_data with ratio = {:.0%}'.format(test_ratio))
        dataset.start_indices = start_indices[split_idx:]
        logging.info('n_test_data = {}'.format(len(dataset)))
        logging.info('labels of extracted test_data \n{}'.format(eval_dataset(dataset)))

    return dataset


def eval_dataset(dataset):
    labels = [imu['label'] for imu in dataset]

    lbl_set = set(labels)

    lbl_count = {label : 0 for label in lbl_set}
    for label in labels:
        lbl_count[label] += 1 
    return lbl_count


def read_config():
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    return config

def load_checkpoint(model, path_to_data, file_name, device_id):
    path_to_checkpoint = file_name if os.path.isfile(file_name) else path_to_data + 'checkpoints/' + file_name
    if os.path.isfile(path_to_checkpoint):
        logging.info('Loading checkpoint from {}'.format(path_to_checkpoint))
        model.load_state_dict(torch.load(path_to_checkpoint, map_location=device_id))
    else:
        raise '[INFO] -- {} is no valid path to a checkpoint file'.format(path_to_checkpoint)

def init_cuda(device_id_cfg):
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = device_id_cfg
    np.random.seed(numpy_seed)
    return torch.device(device_id), device_id

def train(model, device, dataset, config):
    # Set to train mode
        model.train()

        # Set the loss
        loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        logging.info("Start train data preparation")

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir(config.get('path_to_data'),'checkpoints'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        logging.info("Start training")
        for epoch in range(n_epochs):

            for batch_idx, minibatch in enumerate(dataloader):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long)
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
                if batch_idx % n_freq_print == 0:
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

        # Plot the loss function
        #loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        #utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

def test(model, device, dataset, config):
    # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        logging.info("Start test data preparation")
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        metric = []
        logging.info("Start testing")
        accuracy_per_label = np.zeros(config.get("num_classes"))
        count_per_label = np.zeros(config.get("num_classes"))
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):

                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long)

                # Forward pass
                res = model(minibatch)

                # Evaluate and append
                pred_label = torch.argmax(res)
                predicted.append(pred_label.cpu().numpy())
                ground_truth.append( label[0].item())
                curr_metric = (pred_label==label).to(torch.int)
                label_id = label[0].item()
                accuracy_per_label[label_id] += curr_metric.item()
                count_per_label[label_id] += 1
                metric.append(curr_metric.item())

        # Record overall statistics
        stats_msg = "Performance of tmp1 on tmp2"
        confusion_mat = confusion_matrix(ground_truth, predicted, labels = list(range(config.get("num_classes"))))
        print(confusion_mat.shape)
        stats_msg = stats_msg + "\n\tAccuracy: {:.3f}".format(np.mean(metric))
        accuracies = []
        for i in range(len(accuracy_per_label)):
                print("Performance for class [{}] - accuracy {:.3f}".format(i, accuracy_per_label[i]/count_per_label[i]))
                accuracies.append(accuracy_per_label[i]/count_per_label[i])
        # save dump
        utils.create_output_dir(config.get('path_to_data'), 'test_results')
        np.savez(join(config.get('path_to_data'), 'test_results/') + "_test_results_dump", confusion_mat = confusion_mat, accuracies = accuracies, count_per_label=count_per_label, total_acc = np.mean(metric))
        logging.info(stats_msg)

        # Logging Results
        logging.info('Count_per_label = {}'.format(count_per_label))
        logging.info('Confusion-Matrix:\n{}'.format(confusion_mat))


if __name__ == '__main__':
    main()





