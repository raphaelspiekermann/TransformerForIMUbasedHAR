import matplotlib.pyplot as plt
import numpy as np
from data.dataloader import load_data
import torch

def trans_win():
    features, labels, _, label_dict = load_data('E:\\transformer_dir', 'lara')
    labels = labels.flatten()
    start_idcs = []

    for idx in range(labels.shape[0]-399):
        fst_elem = label_dict[labels[idx]]
        left_middle = label_dict[labels[idx+199]]
        right_middle = label_dict[labels[idx+200]]
        lst_elem = label_dict[labels[idx+399]]
        
        if fst_elem=='Standing' and left_middle=='Standing' and right_middle=='Walking' and lst_elem=='Walking':
            start_idcs.append(idx)


    start_idx = start_idcs[39]
    window = features[start_idx:start_idx+400, 24:27]

    x_values = np.array(range(0, 4000, 10))

    #fig = plt.figure(figsize=(16, 9), dpi=120)
    fig = plt.figure()
    plt.plot(x_values / 1000, window, label=['Acc_X','Acc_Y','Acc_Z'])
    plt.axis([0, 4, -2.5, 2.5])
    plt.legend(loc='upper left')
    plt.xlabel(r'Zeit $[s]$')
    plt.ylabel(r'Beschleunigung $[m/s^2]$')
    plt.savefig('C:\\Users\\Raphael\\Desktop\\transition_window.pdf')
    plt.show()
    plt.close(fig)
    
    
def trans_win_normalized():
    features, labels, _, label_dict = load_data('E:\\transformer_dir', 'lara')
    labels = labels.flatten()
    start_idcs = []

    features = features[:, 24:27]
    
    for idx in range(labels.shape[0]-399):
        fst_elem = label_dict[labels[idx]]
        left_middle = label_dict[labels[idx+199]]
        right_middle = label_dict[labels[idx+200]]
        lst_elem = label_dict[labels[idx+399]]
        
        if fst_elem=='Standing' and left_middle=='Standing' and right_middle=='Walking' and lst_elem=='Walking':
            start_idcs.append(idx)


    start_idx = start_idcs[39]
    imu_data = features[start_idx:start_idx+400]
    print(imu_data.shape)
    print(np.mean(features, axis=0))
    print(np.std(features, axis=0))
    
    imu_data = (imu_data - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    print(imu_data.shape)
    imu_data += np.random.RandomState(seed=42).normal(0, 0.01, imu_data.shape)
    print(imu_data.shape)

    x_values = np.array(range(0, 4000, 10))

    #fig = plt.figure(figsize=(16, 9), dpi=120)
    fig = plt.figure()
    plt.plot(x_values / 1000, imu_data, label=['Acc_X','Acc_Y','Acc_Z'])
    #plt.axis([0, 4, -2.5, 2.5])
    plt.legend(loc='upper left')
    plt.xlabel(r'Zeit $[s]$')
    plt.ylabel(r'Beschleunigung $[m/s^2]$')
    plt.savefig('C:\\Users\\Raphael\\Desktop\\transition_window___.pdf')
    plt.show()
    plt.close(fig)
    

def win_comp():
    features, labels, _, label_dict = load_data('E:\\transformer_dir', 'motionsense')
    labels = labels.flatten()

    walking_idcs = []
    jogging_idcs = []

    for idx in range(labels.shape[0]-199):
        fst_elem = label_dict[labels[idx]]
        lst_elem = label_dict[labels[idx+199]]
            
        if fst_elem == lst_elem == 'wlk':
            walking_idcs.append(idx)
        if fst_elem == lst_elem == 'jog':
            jogging_idcs.append(idx)

    walking_start_idx = walking_idcs[20]
    jogging_start_idx = jogging_idcs[20]

    walking_window = features[walking_start_idx : walking_start_idx + 200, :3]
    jogging_window = features[jogging_start_idx : jogging_start_idx + 200, :3]


    fig, ax = plt.subplots(3,2, layout='constrained')

    x_values = np.array(range(0, 4000, 20))
    achsen = ['X', 'Y', 'Z']
    farben = ['C0', 'C1', 'C2']
    for idx in range(ax.shape[0]):
        win1 = walking_window[:, idx]
        win2 = jogging_window[:, idx]
        
        ax1 = ax[idx][0]
        ax2 = ax[idx][1]
        
        ax1.plot(x_values / 1000, win1, label='Gehen', color=farben[idx])
        ax2.plot(x_values / 1000, win2, label='Jogging', color=farben[idx])
        
        if idx==0:
            ax1.set_title('Gehen')
            ax2.set_title('Joggen')
        if idx==2:    
            ax1.set_xlabel('Zeit [s]')
            ax2.set_xlabel('Zeit [s]')
        
        ax1.set_ylim([-4, 4])
        ax2.set_ylim([-4, 4])
        ax1.set_ylabel(r'Acc$_{}$ $[m/s^2]$'.format(achsen[idx]))
        #ax1.legend(loc='upper left')
        

    plt.savefig('C:\\Users\\Raphael\\Desktop\\walking_vs_jogging.pdf')
    plt.show()
    plt.close(fig)
    
 
def win_comp2():
    features, labels, _, label_dict = load_data('E:\\transformer_dir', 'motionsense')
    labels = labels.flatten()

    walking_idcs = []
    jogging_idcs = []

    for idx in range(labels.shape[0]-199):
        fst_elem = label_dict[labels[idx]]
        lst_elem = label_dict[labels[idx+199]]
            
        if fst_elem == lst_elem == 'wlk':
            walking_idcs.append(idx)
        if fst_elem == lst_elem == 'jog':
            jogging_idcs.append(idx)

    walking_start_idx = walking_idcs[20]
    jogging_start_idx = jogging_idcs[20]

    walking_window = features[walking_start_idx : walking_start_idx + 200, :3]
    jogging_window = features[jogging_start_idx : jogging_start_idx + 200, :3]


    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16, 9), dpi=120, layout='constrained')

    x_values = np.array(range(0, 4000, 20))
    
        
    ax1.plot(x_values / 1000, walking_window, label=['Acc-X', 'Acc-Y', 'Acc-Z'])
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel(r'Beschleunigung $[m/s^2]$')
    ax1.legend(loc='upper left')
    ax1.set_ylim([-4,4])
    
    ax2.plot(x_values / 1000, jogging_window, label=['Acc-X', 'Acc-Y', 'Acc-Z'])
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel(r'Beschleunigung $[m/s^2]$')
    ax2.legend(loc='upper left')
    ax2.set_ylim([-4,4])

    plt.savefig('C:\\Users\\Raphael\\Desktop\\walking_vs_jogging.pdf')
    plt.show()
    plt.close(fig)    
    
    
#trans_win_normalized()
#trans_win()
win_comp()