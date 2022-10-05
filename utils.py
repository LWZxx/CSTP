
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class Compsose_Dataset(Dataset):
    def __init__(self, Data, label):
        super(Compsose_Dataset, self).__init__()
        self.data = torch.tensor(Data.T)
        self.targets = torch.tensor(label.T)

    def __getitem__(self, indx):
        image = self.data[indx]
        target = self.targets[indx]

        return image, target

    def __len__(self):
        return len(self.data)

def relu(x, alpha):
    return np.maximum(0, x + alpha)


def SoftMax(X_I):
    X = X_I - np.max(X_I, axis=0)
    X = np.exp(X)
    X = X / np.sum(X, axis=0)
    return X


def Max_Classifier(Data, Label):
    ind = np.argmax(Data, axis=0)
    if Label.shape[0] != 1:
        Label = np.argmax(Label, axis=0)

    return np.sum(ind == Label) / Label.size

def Back_norm_gd(Xd2, W, Xd1, Label, beta):
    A = np.matmul(W.T, Xd2 - Label)
    r = np.sum(np.abs(np.diag(Xd1))) / np.sum(np.abs(np.diag(A)))

    Xd1 = Xd1 - beta * r * A
    return Xd1

def Hard_Thresholod(Data, k = 0.1):
    A = np.zeros_like(Data)
    (w, h) = Data.shape
    num_k = int(k * w)
    inds = np.argsort(-(Data), axis=0)

    for j in range(h):
        A[inds[:num_k, j], j] = Data[inds[:num_k, j], j]

    return A

def Forward_Propagation(Net_work, X_Train, L_size):
    Result = {}
    Result["Y0"] = X_Train
    for i in range(L_size - 2):
        Result["Xd" + str(i + 1)] = relu(np.matmul(Net_work["D" + str(i + 1)], Result["Y" + str(i)]),
                                         Net_work["alpha" + str(i + 1)])
        Result["Y" + str(i + 1)] = np.matmul(Net_work["W" + str(i + 1)], Result["Xd" + str(i + 1)])
    Result["Y" + str(L_size - 1)] = np.matmul(Net_work["W" + str(L_size - 1)], Result["Y" + str(L_size - 2)]) + \
                                    Net_work["b" + str(L_size - 1)].reshape(-1, 1)
    Result["Z" + str(L_size - 1)] = SoftMax(Result["Y" + str(L_size - 1)])
    return Result


def Back_inv(W, Y, ratio=1.0):
    I = np.sqrt(np.sum(W ** 2, axis=0))
    W = W / I[None, :]
    I = np.diag(I)
    X_N = np.matmul(np.matmul(np.linalg.inv(I), W.T), Y)
    X_N = Hard_Thresholod(X_N, ratio)

    return X_N


def Backward_Propagation(Network, Result, L_size, Train_Labels_OH, beta=0.24, ratio=1.0):
    Target = {}
    Target["Z0"] = Train_Labels_OH
    Target["Y" + str(L_size - 2) + "_N"] = Back_norm_gd(Result["Z" + str(L_size - 1)], Network["W" + str(L_size - 1)],
                                                        Result["Y" + str(L_size - 2)], Train_Labels_OH, beta)
    print(Max_Classifier(np.matmul(Network["W" + str(L_size - 1)], Target["Y" + str(L_size - 2) + "_N"]) +
                         Network["b" + str(L_size - 1)].reshape(-1, 1), Train_Labels_OH))
    for i in range((L_size - 2), 1, -1):
        Target["Xd" + str(i) + "_N"] = Back_inv(Network["W" + str(i)], Target["Y" + str(i) + "_N"],ratio)
        Target["Y" + str(i - 1) + "_N"] = np.matmul(Network["D" + str(i)].T, Target["Xd" + str(i) + "_N"])

    return Target


