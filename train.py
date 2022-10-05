import sys
import torch.random

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import *
from utils import *

import numpy as np
import torch.nn as nn
import torch

def update_Net_Local(F_Train, F_Test, Network, i, Target, Test_Labels, L_Num, update_set, best_acc):
    set = update_set["Local_optim"]
    epoch, batch = set["epoch"], set["batch_size"]
    device = update_set["device"]

    D, W = Network["D" + str(i)], Network["W" + str(i)]
    net = Net_Local(D, W).to(device)
    mse_loss = nn.MSELoss().to(device)

    Train_Labels = Target["Y" + str(i) + "_N"]

    train_data = Compsose_Dataset(F_Train, Train_Labels)
    train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=batch, shuffle=True)
    optimizer = set["type"](net.parameters(), **(set["optim_params"]))
    net.train()

    for j in range(0, epoch):
        with tqdm(iterable=train_loader) as t:
            running_results = {'batch_sizes': 0, 'loss': 0, }
            for img, target in train_loader:
                batch_size = img.size(0)
                if batch_size <= 0:
                    continue
                running_results['batch_sizes'] += batch_size

                img = img.to(device)
                target = target.to(device)
                output = net(img)
                optimizer.zero_grad()
                loss = mse_loss(output, target)
                loss.backward()
                optimizer.step()

                running_results['loss'] += loss.item() * batch_size

                t.set_description(desc='[%d] Loss: %.4f lr: %.7f' % (
                    j, running_results['loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
                t.update()

        Network["D" + str(i)], Network["W" + str(i)] = net.D.weight.data.cpu().numpy(), net.W.weight.data.cpu().numpy()
        Network["alpha" + str(i)] = net.activation.bias.data.cpu().numpy()
        Result = Forward_Propagation(Network, F_Test, L_Num)
        Acc = Max_Classifier(Result["Z" + str(L_Num - 1)], Test_Labels)
        print("Test Acc: %.4f" % Acc, file=sys.stderr)

        if Acc > best_acc:
            best_acc = Acc

    return Network, best_acc


def update_Net_Classifier(F_Train, F_Test, Network, i, Train_Labels, Test_Labels, L_num, update_set, best_acc):

    set = update_set["Classifier_optim"]
    epoch, batch = set["epoch"], set["batch_size"]
    device = update_set["device"]

    W, b = Network["W" + str(i)], Network["b" + str(i)]

    net = Net_Classifier(W, b).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    train_data = Compsose_Dataset(F_Train, Train_Labels)
    train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=batch, shuffle=True)
    optimizer = set["type"](net.parameters(), **(set["optim_params"]))
    net.train()

    for j in range(0, epoch):
        with tqdm(iterable=train_loader) as t:
            running_results = {'batch_sizes': 0, 'loss': 0, }
            for img, target in train_loader:
                batch_size = img.size(0)
                if batch_size <= 0:
                    continue
                running_results['batch_sizes'] += batch_size

                img = img.to(device)
                target = target.long().squeeze(1).to(device)
                output = net(img)
                optimizer.zero_grad()
                loss = ce_loss(output, target)
                loss.backward()
                optimizer.step()

                running_results['loss'] += loss.item() * batch_size

                t.set_description(desc='[%d] Loss: %.4f lr: %.7f' % (
                    j, running_results['loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
                t.update()
        Network["W" + str(i)] = net.W.weight.data.cpu().numpy()
        Network["b" + str(i)] = net.W.bias.data.cpu().numpy()

        Result = Forward_Propagation(Network, F_Test, L_num)
        Acc = Max_Classifier(Result["Z" + str(L_num - 1)], Test_Labels)
        print("Test Acc: %.4f" % Acc, file=sys.stderr)

        if Acc > best_acc:
            best_acc = Acc

    return Network, best_acc


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    #set_seed(0)

    config = {
            "Dim": np.array([3072, 3072, 2048, 1024, 10]),
            "Local_optim": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 0},
                               "epoch": 7, "batch_size": 64},
            "Classifier_optim": {"type": optim.Adam, "optim_params": {"lr": 1e-3, "weight_decay": 0},
                             "epoch": 20, "batch_size": 64},
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "beta": 4.0,
            "Backward_ratio": 0.25,
    }
    depth = config["Dim"].shape[0]

    ########################
    #  Prepare Data
    ########################
    data = np.load('./data/Cifar10.npz')

    X_Train = data['X_Train']
    X_Test = data['X_Test']
    Train_Labels = data['Train_Labels']
    Test_Labels = data['Test_Labels']
    Train_Labels_OH = data['Train_Labels_OH']
    Test_Labels_OH = data['Test_Labels_OH']
    del data

    u1 = np.mean(X_Train, axis=1)
    std1 = np.std(X_Train, axis=1)

    X_Train = (X_Train - u1[:, None]) / std1[:, None]
    X_Test = (X_Test - u1[:, None]) / std1[:, None]

    ######  Initialization
    Network = Model_Initialization(config["Dim"], depth)
    ######  Forward Propagation
    Result = Forward_Propagation(Network, X_Train, depth)

    Acc = Max_Classifier(Result["Z" + str(depth - 1)], Train_Labels)
    print("Initialization_Acc: ", Acc)
    ######  Backward Propagation
    Target = Backward_Propagation(Network, Result, depth, Train_Labels_OH, beta = config["beta"], ratio = config["Backward_ratio"])
    ########################
    #  Train
    ########################
    Best_Acc = 0
    for i in range(1, depth - 1):
        Network, Best_Acc = update_Net_Local(Result["Y" + str(i - 1)], X_Test, Network, i, Target, Test_Labels,
                                            depth, config, Best_Acc)
        Result = Forward_Propagation(Network, X_Train, depth)

    Network, Best_Acc = update_Net_Classifier(Result["Y" + str(depth - 2)], X_Test, Network, depth - 1,
                                            Train_Labels, Test_Labels, depth, config, Best_Acc)
    print(Best_Acc)
