import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import os
import sklearn.metrics
import math
from contextlib import closing
import sqlite3
import time
from datetime import datetime

class Evaluator():
    def __init__(self, input_path, data_path, output_path=None):
        if os.path.isdir(input_path):
            self.__path__ = input_path
            out_ex = output_path is not None and os.path.isdir(output_path)
            self.__out_path__ = output_path if out_ex else None
        else:
            raise ValueError('path = {} is not valid for validation'.format(input_path))
        if os.path.isdir(data_path):
            self.__data_path__ = data_path
        else:
            raise ValueError('data_path = {} does not exist'.format(data_path))
        
    def create_database(self):
        db_path = join(self.__out_path__, 'database.db') 
        if os.path.isfile(db_path):
            os.remove(db_path)
        with closing(sqlite3.connect(db_path)) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute('CREATE TABLE Experiment (MODEL TEXT, WINDOW INTEGER, NORMALIZATION INTEGER, SPLIT_TYPE TEXT, TORCH_SEED INTEGER, ACCURACY REAL, F1_SCORE REAL, BEST_EPOCH INTEGER, TRAIN_MINUTES INTEGER, PARAMETERS INTEGER)')
                classifications, losses, configs, _, run_names = get_runs(self.__path__)
                for classification, loss, cfg, run_name in zip(classifications, losses, configs, run_names):
                    model = cfg['data']['model_name']
                    win_size = cfg['data']['window_size']
                    normalization = int(cfg['data']['normalize'])
                    split_type = cfg['data']['split_type']
                    seed = cfg['setup']['torch_seed']   
                    
                    # accuracy and f1
                    acc = sklearn.metrics.accuracy_score(y_true=classification[0], y_pred=classification[1]) * 100
                    f1 = sklearn.metrics.f1_score(y_true=classification[0], y_pred=classification[1], average='weighted') * 100
                    
                    # best_epoch
                    best_epoch = np.argmax(loss[2])
                    n_params = -1
                    # train_Time and paramaters
                    with open(join(self.__path__, run_name, '{}.log'.format(run_name)), mode='r') as f:
                        txt = f.read()
                        lines = txt.split('\n')
                        fst_epoch_line = ''                      
                        lst_epoch_line = ''
                        for line in lines:
                            if line.find('Epoch[') != -1 and fst_epoch_line == '':
                                fst_epoch_line = line
                            if line.find('Epoch[') != -1:
                                lst_epoch_line = line
                            if line.find('parameters') != -1:
                                n_params = int(line.split(' ')[-2])
                        
                        start_time = fst_epoch_line.split(']')[0][1:]
                        end_time = lst_epoch_line.split(']')[0][1:]
                        
                        time_format = '%Y-%m-%d %H:%M:%S'
                        start_time = datetime.strptime(start_time, time_format)
                        end_time = datetime.strptime(end_time, time_format)
                        
                        time_delta = end_time - start_time
                        time_delta = round(time_delta.total_seconds() / 60) 
                    
                    
                    
                    ins_str = "INSERT INTO Experiment VALUES ('{}', {}, {}, '{}', {}, {:.2f}, {:.2f}, {}, {}, {})".format(model, win_size, normalization, split_type, seed, acc, f1, best_epoch, time_delta, n_params)
                    #print(ins_str)
                    cursor.execute(ins_str)
                    
                connection.commit()
                
    def loss_progress(self):
        loss_progress(self.__path__, self.__out_path__)

    def validation_scores(self):
        validation_scores(self.__path__)
        
    def avg_accuracies(self):
        avg_accuracies(self.__path__)

    def get_top_performing_configs(self, n=5):
        get_top_performing_configs(self.__path__, n)
    
    def eval_models(self):
        eval_models(self.__path__, self.__out_path__)
        
    def avg_best_epochs_per_model(self, group_by_window_size=False):
        avg_best_epochs_per_model(self.__path__, group_by_window_size)
    
    
# UTILITY FUNCTIONS
def plot_class_distr(labels, label_dict=None, path=None):
    if path is not None:
        path = join(path, 'class_distribution.png')
    
    assert labels.ndim == 1
    
    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                    '%d' % float(height),
                    ha='center', va='bottom')


    fig, ax = plt.subplots()
        
    if label_dict is not None:
        labels = [label_dict[label] for label in labels]
        
    labels_unique = np.unique(labels, axis=None) if isinstance (labels, np.ndarray) else list(set(labels))
    labels_count = [0] * len(labels_unique)
    
    for idx, label1  in enumerate(labels_unique):
        for label2 in labels:
            labels_count[idx] += int(label1 == label2)
    
    y_values = np.arange(len(labels_unique))

    plt.title('N = {:_}'.format(len(labels)))

    c = ['b', 'g', 'r', 'c', 'm', '#F97306', 'y', 'k', 'tab:gray']
    c_rects = plt.bar(y_values, labels_count, align='center', alpha=0.66, color=c)

    ax.set_xticks(range(len(labels_unique)))
    ax.set_xticklabels(labels_unique, rotation='vertical')

    autolabel(c_rects)

    plt.ylim(0, 400000)
    
    if path != None:
        plt.savefig(path)
        
    plt.show()
    
    plt.close('all')
    matplotlib.rc_file_defaults()


def plot(data, x_vals = None, title=None, x_title=None, y_titles=None, path=None):
    if x_vals is None:
        x_vals = np.array(range(len(data[0])))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for idx in range(len(data)):        
        if y_titles is not None:
            plt.plot(x_vals, data[idx], label=y_titles[idx])
        else:
            plt.plot(x_vals, data[idx])
    plt.grid()
    if y_titles is not None:
        plt.legend(loc='upper left')
    if x_title is not None:
        plt.xlabel(x_title)
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()
    
    plt.close(fig)
    matplotlib.rc_file_defaults()


def scatter(data, x_vals = None, title=None, x_title=None, y_title=None, y_titles=None, path=None):
    if x_vals is None:
        x_vals = np.array(range(len(data[0])))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for idx in range(len(data)):        
        if y_titles is not None:
            plt.scatter(x_vals, data[idx], label=y_titles[idx])
        else:
            plt.scatter(x_vals, data[idx])
    plt.grid()
    if y_titles is not None:
        plt.legend(loc='upper left')
    if x_title is not None:
        plt.xlabel(x_title)
    if y_title is not None:
        plt.ylabel(y_title)
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()
    
    plt.close(fig)
    matplotlib.rc_file_defaults()


def create_heatmap(real_labels, pred_labels, labels, label_dict=None, title=None, file_name=None , normalize=True):

    if label_dict != None:
        real_labels = [label_dict[label] for label in real_labels]
        pred_labels = [label_dict[label] for label in pred_labels]
        labels = [label_dict[label] for label in labels]
    
    confusion_matr = sklearn.metrics.confusion_matrix(y_true=real_labels, y_pred=pred_labels, labels=labels)
    if normalize:
        confusion_matr = np.array(confusion_matr, dtype=np.float64)
        for row in confusion_matr:
            row *= 100 / np.sum(row)
    
    fig = plt.figure(figsize=(16, 9), dpi=120)
    sns.set_theme()
    own_cmap = sns.color_palette("viridis", as_cmap=True) #sns.color_palette("pastel", as_cmap=True)
    
    ax = sns.heatmap(confusion_matr, annot=True, fmt=".2f" if normalize else "d", cmap=own_cmap, cbar=False,
                     linewidths=.5, xticklabels=labels, yticklabels=labels)
    if normalize:
        for t in ax.texts: t.set_text(t.get_text() + "%")
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    
    plt.xlabel("Predicted Labels",rotation=0, fontsize=22)
    plt.ylabel("Real Labels", rotation=90, fontsize=22)
    
    
    ax.set_title(title, fontsize=22)

    if file_name != None:
        plt.savefig(file_name)
        
    plt.close(fig)
    
    matplotlib.rc_file_defaults()


def get_files(path, file_extension):
    file_names = []
    for root, _, files in os.walk(path, topdown=False):
        for file_name in files:
            if file_name.endswith(file_extension):
                file_names.append(join(root, file_name))
    return file_names


def get_runs(path):
    classification_files = get_files(path, 'classifications.npy')
    loss_files = get_files(path, 'loss.npy')
    config_files = get_files(path, 'config.json')

    for cl_file, loss_file, cfg_file in zip(classification_files, loss_files, config_files):
        x = ''.join(cl_file.split('_')[:-1])
        y = ''.join(loss_file.split('_')[:-1])
        z = ''.join(cfg_file.split('_')[:-1])
        assert x == y and y == z

    classifications = [np.load(classifications) for classifications in classification_files]
    losses = [np.load(loss) for loss in loss_files]
    configs = []
    models = []
    run_names = []
       
    
    for cfg in config_files:
        c = cfg.split('\\')[-1]
        c = '_'.join(c.split('_')[:-1])
        run_names.append(c)
        with open(cfg, "r") as read_file:
            conf = json.load(read_file)
            # Renaming model_name for simplicity
            conf['data']['model_name'] = translate_model_name(conf['data']['model_name'])
            configs.append(conf)
            if conf['data']['model_name'] not in models:
                models.append(conf['data']['model_name'])
    
    assert len(classification_files) == len(losses) and len(losses) == len(configs)
    
    model_dict = {m: pos for m, pos in zip(models, range(len(models)))}

    return classifications, losses, configs, model_dict, run_names


def translate_model_name(m_name):
    if m_name.find('small') != -1:
        return 'small_transformer'
    if m_name.find('medium') != -1 or m_name == 'transformer':
        return 'default_transformer'
    if m_name.find('big') != -1:
        return 'big_transformer'
    if m_name.find('lara') != -1:
        return 'lara_tcnn'  
    if m_name.find('raw') != -1:
        return 'raw_transformer'
    return m_name


## EVALATUATION FUNCTIONS

def loss_progress(path, output_path=None):
    _, losses, configs, model_dict, _ = get_runs(path)
    models = list(model_dict)

    shortest_run = math.inf
    for l in losses:
        shortest_run = min(shortest_run, l.shape[1]) 
    
    for model in models:
        output_path_ = join(output_path, 'eval_losses[{}].png'.format(model)) if output_path is not None else None

        model_loss = [loss[:, :shortest_run] for loss, cfg in zip(losses, configs) if cfg['data']['model_name'] == model]
        model_loss = np.array(model_loss)

        # Averaging
        model_loss = np.average(model_loss, axis=0)
    
        # Printing average-accuracies
        avg_acc = np.average(model_loss[-1]).item()
        min_acc = np.min(model_loss[-1]).item()
        max_acc = np.max(model_loss[-1]).item()
    
        eval_str = '{}: Min_Accuracy {:.3f} | Avg_Accuracy {:.3f} | Max_Accuracy {:.3f}'.format(model, min_acc, avg_acc, max_acc)
        titles = ['Loss_avg', 'Val_Loss_avg', 'Val_acc']
        plot(model_loss, title=eval_str, x_title='Epochs', y_titles=titles, path=output_path_)


def avg_accuracies(path):
    classifications, _, configs, model_dict, _ = get_runs(path)
    
    models = []
    window_sizes = []
    normalizations = []
    
    for cfg in configs:
        if cfg['data']['model_name'] not in models:
            models.append(cfg['data']['model_name'])
        if cfg['data']['window_size'] not in window_sizes:
            window_sizes.append(cfg['data']['window_size'])
        if cfg['data']['normalize'] not in normalizations:
            normalizations.append(cfg['data']['normalize'])
    
    accuracies = []
    
    for c in classifications:
        acc = sklearn.metrics.accuracy_score(y_true=c[0], y_pred=c[1])
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    
    model_dict = {model_name: pos for pos, model_name in enumerate(models)}
    window_dict = {win: pos for pos, win in enumerate(window_sizes)}
    normalization_dict = {n: pos for pos, n in enumerate(normalizations)}
    
    N = len(model_dict) * len(window_dict) * len(normalization_dict)
    aggr_ls = [[] for _ in range(N)]
    
    for cfg, acc in zip (configs, accuracies):
        m = model_dict[cfg['data']['model_name']]
        w = window_dict[cfg['data']['window_size']]
        n = normalization_dict[cfg['data']['normalize']]
        
        m_shift = N // len(models)
        n_shift = N // (len(models) * len(normalizations))
        w_shift = N // (len(models) * len(normalizations) * len(window_sizes)) 
        
        idx = m * m_shift +  n * n_shift + w * w_shift
        
        aggr_ls[idx].append(acc)
    
    aggr_ls = np.array(aggr_ls)
    
    f_str = '|{:^20}|{:^13}|{:^13}|{:^19}|'
    split_line = f_str.format('-'*20, '-'*13,'-'*13,'-'*19)
    
    # Printing results
    print(split_line)
    print(f_str.format('Model', 'Normalize', 'Window_Size', 'Accuracy [%]'))
    print(split_line)
    
    for m in models:
        for n in normalizations:
            for w in window_sizes:
                m_shift = N // len(models)
                n_shift = N // (len(models) * len(normalizations))
                w_shift = N // (len(models) * len(normalizations) * len(window_sizes)) 

                idx = model_dict[m] * m_shift + window_dict[w] * w_shift + normalization_dict[n] * n_shift
                
                avg = np.mean(aggr_ls[idx])
                std = np.std(aggr_ls[idx])
                
                acc = '{:.2f} +-{:.2f}'.format(avg * 100, std * 100)
                print(f_str.format(m,n,w,acc))
    print(split_line)


def validation_scores(path):
    classifications, _, configs, model_dict, _ = get_runs(path)
    model_names = list(model_dict.keys())
    
    f_str = '|{:^20}|{:^20}|{:^20}|{:^20}|'
    split_line = f_str.format('-'*20, '-'*20,'-'*20,'-'*20)
    
    # Printing results
    print(split_line)
    print(f_str.format('Model', 'Min: Acc / F1 [%]', 'Avg: Acc / F1 [%]', 'Max: Acc / F1 [%]'))
    print(split_line)
    
    
    for model_name in model_names:
        # Preperation
        classifications_model = [c for cfg, c in zip(configs, classifications) if cfg['data']['model_name'] == model_name]
        accuracies = []
        f1_scores = []
    
        for c in classifications_model:
            acc = sklearn.metrics.accuracy_score(y_true=c[0], y_pred=c[1])
            f1 = sklearn.metrics.f1_score(y_true=c[0], y_pred=c[1], average='weighted')
            accuracies.append(acc)
            f1_scores.append(f1)

        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        
        s = '{:.2f} / {:.2f}'
        min_ = s.format(np.min(accuracies) * 100, np.min(f1_scores) * 100)
        avg_ = s.format(np.mean(accuracies) * 100, np.mean(f1_scores) * 100)
        max_ = s.format(np.max(accuracies) * 100, np.max(accuracies) * 100)
        
        print(f_str.format(model_name, min_, avg_, max_))
    print(split_line)
  
    
def get_top_performing_configs(path, n):
    classifications, _, configs, _, run_names = get_runs(path)
    
    accuracies = []
    
    for c in classifications:
        acc = sklearn.metrics.accuracy_score(y_true=c[0], y_pred=c[1])
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    indices = np.argsort(accuracies)
    
    
    format_str = '|{:^23}|{:^15}|{:^20}|{:^11}|{:^13}|{:^15}|{:^10}|'
    split_line = format_str.format('-' * 23, '-' * 15, '-' * 20, '-' * 11, '-' * 13, '-' * 15,'-' * 10,)
    header = format_str.format('Run', 'Accuracy [%]', 'Model', 'Normalize', 'Window', 'Split_Type', 'Seed')
    
    print(split_line)
    print(header)
    print(split_line)
    
    for i in range(-1, -(n+1), -1):
        idx = indices[i]
        name = run_names[idx]
        acc = '{:.2f}'.format(accuracies[idx].item() * 100)
        cfg = configs[idx]
        model = cfg['data']['model_name']
        normalize = cfg['data']['normalize']
        win = cfg['data']['window_size']
        split = 'P' if cfg['data']['split_type'] == 'person' else 'PR'
        seed = cfg['setup']['torch_seed']
        string = format_str.format(name, acc, model, normalize, win, split, seed)
        print(string)
        
    print(split_line)


def eval_models(path, output_path=None):
    if output_path is not None:
        output_path = join(output_path, 'eval_models.png')
    
    classifications, _, configs, model_dict, _ = get_runs(path)
    
    accuracies = []
    models = []
    models_unique = []
    window_sizes = []
    windows_unique = []
    
    for idx, classification in enumerate(classifications):
        cfg = configs[idx]
        
        acc = sklearn.metrics.accuracy_score(y_true=classification[0], y_pred=classification[1])
        accuracies.append(acc)
        
        model = cfg['data']['model_name']
        models.append(model)
        if model not in models_unique:
            models_unique.append(model)
        
        win = cfg['data']['window_size']
        window_sizes.append(win)
        if win not in windows_unique:
            windows_unique.append(win)
    
    
    model_dict = {model_name: pos for model_name, pos in zip(models_unique, range(len(models_unique)))}
    window_dict = {win: pos for win, pos in zip(windows_unique, range(len(windows_unique)))}
    
    aggr_array = np.zeros(shape=(len(model_dict), len(window_dict), 2), dtype=np.float32)
    
    for m, w, a in zip(models, window_sizes, accuracies):
        i = model_dict[m]
        j = window_dict[w]
        aggr_array[i,j,0] += 1
        aggr_array[i,j,1] += a
    
    y_titles = []
    y_values = []
        
    for m in models_unique:
        ls = []
        for w in windows_unique:
            m_idx = model_dict[m]
            w_idx = window_dict[w]
            
            cnt = aggr_array[m_idx,w_idx,0]
            acc = aggr_array[m_idx,w_idx,1]
            ls.append(acc / cnt)
        y_values.append(ls)
        y_titles.append(m)
        
    x_values = windows_unique
    scatter(y_values, x_vals = x_values, title='', x_title='Window_Sizes', y_titles=y_titles, path=output_path)    
    

def avg_best_epochs_per_model(path, group_by_window_size):
    _, losses, configs, model_dict, _ = get_runs(path)
    if group_by_window_size:
        
        window_sizes = [cfg['data']['window_size'] for cfg in configs]
        window_sizes = sorted(list(set(window_sizes)))
        N_windows = len(window_sizes)
        window_idx_dict = {win_size: idx for win_size, idx in zip(window_sizes, range(N_windows))}
        print(window_sizes)
        
        N = len(model_dict)
        
        best_epochs = [[] for _ in range(N * N_windows)]
        for loss, cfg in zip(losses, configs):
            best_epoch = np.argmax(loss[2])
            model = cfg['data']['model_name']
            win_size = cfg['data']['window_size']
            idx = N_windows * model_dict[model] + window_idx_dict[win_size]
            best_epochs[idx].append(best_epoch)
        best_epochs = np.array(best_epochs)
        
        f_str = '|{:^20}|{:^15}|{:^13}|{:^13}|{:^13}|'
        split_line = f_str.format('-'*20, '-'*15, '-'*13,'-'*13,'-'*13)
        
        # Printing results
        print(split_line)
        print(f_str.format('Model', 'Window', 'Min', 'Avg', 'Max'))
        print(split_line)
        
        for model, idx1 in model_dict.items():
            for win_size, idx2 in window_idx_dict.items():
                idx = N_windows * idx1 + idx2
                win_size
                min_ = '{:d}'.format(int(np.min(best_epochs[idx])))
                avg_ = '{:.2f}'.format(np.mean(best_epochs[idx]))
                max_ = '{:d}'.format(int(np.max(best_epochs[idx])))
                print(f_str.format(model, win_size, min_, avg_, max_))
        print(split_line)
    else:
        N = len(model_dict)
        best_epochs = [[] for _ in range(N)]
        for loss, cfg in zip(losses, configs):
            best_epoch = np.argmax(loss[2])
            model = cfg['data']['model_name']
            idx = model_dict[model]
            best_epochs[idx].append(best_epoch)
        best_epochs = np.array(best_epochs)
        
        f_str = '|{:^20}|{:^13}|{:^13}|{:^13}|'
        split_line = f_str.format('-'*20, '-'*13,'-'*13,'-'*13)
        
        # Printing results
        print(split_line)
        print(f_str.format('Model', 'Min', 'Avg', 'Max'))
        print(split_line)
        
        for model, idx in model_dict.items():
            min_ = '{:d}'.format(int(np.min(best_epochs[idx])))
            avg_ = '{:.2f}'.format(np.mean(best_epochs[idx]))
            max_ = '{:d}'.format(int(np.max(best_epochs[idx])))
            print(f_str.format(model, min_, avg_, max_))
        print(split_line)
                
        
        
def main():
    path = 'C:\\Users\\Raphael\\Desktop\\Experiment2\\runs'
    output_path = 'C:\\Users\\Raphael\\Desktop\\Experiment2'
    evaluator = Evaluator(path, 'C:', output_path)
    evaluator.create_database()
    
    #evaluator.loss_progress()
    #evaluator.validation_scores()
    #evaluator.avg_accuracies()
    #evaluator.get_top_performing_configs()
    #evaluator.eval_models()
    #evaluator.avg_best_epochs_per_model(group_by_window_size=True)


    
    

if __name__ == '__main__':
    main()