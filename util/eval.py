import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import os
#from .dataloader import load_data, IMUDataset
import sklearn.metrics
import scipy
import math


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
        
    def loss_progress(self):
        loss_progress(self.__path__, self.__out_path__)


    def validation_scores(self):
        validation_scores(self.__path__)



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
    
    confusion_matr = confusion_matrix(y_true=real_labels, y_pred=pred_labels, labels=labels)
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

    classifications = [np.load(classifications) for classifications in classification_files]
    losses = [np.load(loss) for loss in loss_files]
    configs = []
    models = []
    
    for cfg in config_files:
        with open(cfg, "r") as read_file:
            conf = json.load(read_file)
            configs.append(conf)
            if conf['data']['model_name'] not in models:
                models.append(conf['data']['model_name'])

    model_dict = {m: pos for m, pos in zip(models, range(len(models)))}

    return classifications, losses, configs, model_dict


def loss_progress(path, output_path=None):
    _, losses, configs, model_dict = get_runs(path)
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
    
        eval_str = '{}_Validationset: Min_Accuracy {:.3f} | Avg_Accuracy {:.3f} | Max_Accuracy {:.3f}'.format(model.capitalize(), min_acc, avg_acc, max_acc)
        titles = ['Loss_avg', 'Val_Loss_avg', 'Val_acc']
        plot(model_loss, title=eval_str, x_title='Epochs', y_titles=titles, path=output_path_)


def validation_scores(path):
    classifications, _, configs, model_dict = get_runs(path)
    models = list(model_dict)
    
    for model in models:
        model_classifications = [c for c, cfg in zip(classifications, configs) if cfg['data']['model_name'] == model]
        accuracies = []
        f1_scores = []
    
        for c in model_classifications:
            acc = sklearn.metrics.accuracy_score(y_true=c[0], y_pred=c[1])
            f1 = sklearn.metrics.f1_score(y_true=c[0], y_pred=c[1], average='weighted')
            accuracies.append(acc)
            f1_scores.append(f1)

        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)

        accuracy_str = 'Validation_Result: {:12s} | Minimum_Accuracy: {:.3f} | Average_Accuracy: {:.3f} | Maximum_Accuracy: {:.3f}'
        f1_str = 'Validation_Result: {:12s} | Minimum_F1-Score: {:.3f} | Average_F1-Score: {:.3f} | Maximum_F1-Score: {:.3f}'
        print(accuracy_str.format(model, np.min(accuracies), np.average(accuracies), np.max(accuracies)))
        print(f1_str.format(model, np.min(f1_scores), np.average(f1_scores), np.max(f1_scores)))


def avg_accuracies(path):
    cfg_files = get_files(path, 'config.json')
    classification_files = get_files(path, 'classifications.npy')
    classifications = [np.load(classifications) for classifications in classification_files]
    
    configs = []
    models = []
    window_sizes = []
    normalizations = []
    
    for cfg in cfg_files:
        with open(cfg, "r") as read_file:
            conf = json.load(read_file)
            configs.append(conf)
            if conf['data']['model_name'] not in models:
                models.append(conf['data']['model_name'])
            if conf['data']['window_size'] not in window_sizes:
                window_sizes.append(conf['data']['window_size'])
            if conf['data']['normalize'] not in normalizations:
                normalizations.append(conf['data']['normalize'])
    
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
    
    f_str = '|{:^15}|{:^13}|{:^13}|{:^19}|'
    split_line = f_str.format('-'*15, '-'*13,'-'*13,'-'*19)
    
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


def get_top_performing_configs(path, n=5):
    cfg_files = get_files(path, 'config.json')
    classification_files = get_files(path, 'classifications.npy')
        
    configs = []
    for cfg in cfg_files:
        with open(cfg, "r") as read_file:
            configs.append(json.load(read_file))
            
    classifications = [np.load(classifications) for classifications in classification_files]
    run_names = []
    
    for c in classification_files:
        c = c.split('\\')[-1]
        c = '_'.join(c.split('_')[:-1])
        run_names.append(c)
    accuracies = []
    
    for c in classifications:
        acc = sklearn.metrics.accuracy_score(y_true=c[0], y_pred=c[1])
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    indices = np.argsort(accuracies)
    
    config_strings = []
    
    format_str = '|{:^23}|{:^15}|{:^13}|{:^11}|{:^13}|{:^15}|{:^10}|'
    split_line = format_str.format('-' * 23, '-' * 15, '-' * 13, '-' * 11, '-' * 13, '-' * 15,'-' * 10,)
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


def eval_window_sizes(path, output_path=None):
    if output_path is not None:
        output_path = join(output_path, 'eval_window_sizes.png')
    
    cfg_files = get_files(path, 'config.json')
    classification_files = get_files(path, 'classifications.npy')
    
    configs = []
    for cfg in cfg_files:
        with open(cfg, "r") as read_file:
            configs.append(json.load(read_file))
            
    classifications = [np.load(classifications) for classifications in classification_files]
    accuracies = []
    models = []
    models_unique = []
    window_sizes = []
    windows_unique = []
    
    for idx in range(len(classifications)):
        classification = classifications[idx]
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


def configs_eq(config_ls):
    for idx in range(1, len(config_ls)):
        cfg1 = config_ls[idx-1]
        cfg2 = config_ls[idx]
        cfg1['data']['model_name'] = ''
        cfg2['data']['model_name'] = ''
        if cfg1['data'] != cfg2['data'] or cfg1['training'] != cfg2['training'] or cfg1['setup'] != cfg2['setup']:            
            return False
    return True


def compare_models(path, output_path=None):    
    
    if output_path is not None:
        output_path = join(output_path, 'compare_models.png')
    
    cfg_files = get_files(path, 'config.json')
    classification_files = get_files(path, 'classifications.npy')
    
    configs = []
    models = []
    for cfg in cfg_files:
        with open(cfg, "r") as read_file:
            conf = json.load(read_file)
            configs.append(conf)
            if conf['data']['model_name'] not in models:
                models.append(conf['data']['model_name'])
                
    model_dict = {m: pos for m, pos in zip(models, range(len(models)))}
    
    classifications = [np.load(classifications) for classifications in classification_files]
    n_models = len(models)
    classifications_per_model = len(classifications) // n_models
    
    y_values = []
    y_titles = []
       
    for idx in range(classifications_per_model):
        cfgs = []
        accuracies = np.zeros(len(models))
        for model in models:
            m_id = model_dict[model]
            effective_idx = m_id * classifications_per_model + idx
                        
            classification = classifications[effective_idx]
            cfgs.append(configs[effective_idx])
            
            acc = sklearn.metrics.accuracy_score(y_true=classification[0], y_pred=classification[1])
            accuracies[m_id] = acc
        
        assert configs_eq(cfgs)
        y_values.append(accuracies)
        
    y_titles = models
    y_values = np.array(y_values).transpose()
    
    scatter(y_values, x_vals=None, title='Model_Comparison', x_title='Setting_ID', y_title='Accuracy', y_titles=y_titles, path=output_path)


def normalize_effect(path):
    cfg_files = get_files(path, 'config.json')
    classification_files = get_files(path, 'classifications.npy')
    classifications = [np.load(classifications) for classifications in classification_files]
    
    configs = []
    for cfg in cfg_files:
        with open(cfg, "r") as read_file:
            configs.append(json.load(read_file))
    
    normalized = []
    unnormalized = []
    
    for idx in range(len(classifications)):
        classification = classifications[idx]
        acc = sklearn.metrics.accuracy_score(y_true=classification[0], y_pred=classification[1])
        
        if configs[idx]['data']['normalize']:
            normalized.append(acc)
        else:
            unnormalized.append(acc)
    
    assert len(normalized) == len(unnormalized)
    
    normalized = np.array(normalized)
    unnormalized = np.array(unnormalized)
    t_test = scipy.stats.ttest_ind(normalized,
                              unnormalized,
                              equal_var=False)
    print(t_test)
    print(scipy.stats.wilcoxon(normalized, unnormalized, alternative='greater'))


def get_data_distribution(path):
    pass



def main():
    path = 'C:\\Users\\Raphael\\Desktop\\Experiment2\\runs'
    output_path = 'C:\\Users\\Raphael\\Desktop\\Experiment2'
    evaluator = Evaluator(path, 'C:', output_path)

    #evaluator.loss_progress()
    evaluator.validation_scores()

main()