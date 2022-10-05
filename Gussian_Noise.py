import sys
import numpy as np
import matplotlib.pyplot as plt

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

def CSTP_Test(Net, F_Test, Len, Test_Labels):
    Result = Forward_Propagation(Net, F_Test, Len)
    Acc = Max_Classifier(Result["Z" + str(Len - 1)], Test_Labels)
    # print("CSTP Test_Acc", Acc)
    return Acc

def BP_Test(Net, F_Test, Len, Test_Labels):
    Result = Forward_Propagation(Net, F_Test, Len)
    Acc = Max_Classifier(Result["Z" + str(Len - 1)], Test_Labels)
    # print("BP Test_Acc", Acc)
    return Acc



def Random_Gussian_All_Noise(F_Test, u1, std1, m ):
    var = 0.1 * m

    temp = F_Test.copy()
    noise = np.random.normal(m, var ** 0.5,temp.shape)
    temp = temp + noise
    temp = np.clip(temp, 0, 255)
    temp = np.uint8(temp)

    temp = temp.reshape([-1,10000], order='F')
    temp = (temp - u1[:, None]) / std1[:, None]

    CSTP_Acc = CSTP_Test(CSTP_Net, temp, depth, Test_Labels)
    BP_Acc = BP_Test(BP_Net, temp, depth, Test_Labels)
    return CSTP_Acc , BP_Acc


def Test_Random_Gussian_All_Noise(Test, Ite, u, std):
    mean = np.arange(0, 100, 10)
    CSTP , BP = np.zeros([Ite, mean.shape[0]]),  np.zeros([Ite, mean.shape[0]]) # [], []
    for i in range(Ite):
        for j in range(mean.shape[0]):
            CSTP_acc, Bp_acc = Random_Gussian_All_Noise(Test, u, std, mean[j])
            CSTP[i,j] = CSTP_acc
            BP[i,j] = Bp_acc
    print("Avg CSTP:", np.sum(CSTP,axis=0)/Ite)
    print("Avg BP:", np.sum(BP,axis=0)/Ite)
    plt.figure()
    plt.plot(mean, np.sum(CSTP,axis=0)/Ite,label='CSTP')
    plt.plot(mean, np.sum(BP,axis=0)/Ite,label='BP')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    BP_Net = np.load('./model/BP_3072-3072-2048-1024-10.npy',allow_pickle=True)[None][0]
    CSTP_Net = np.load('./model/CSTP_3072-3072-2048-1024-10.npy', allow_pickle=True)[None][0]

    Dim = np.array([3072, 3072, 2048, 1024, 10])
    depth = Dim.shape[0]

    data = np.load('./data/Cifar10.npz')
    X_Train = data['X_Train']
    X_Test = data['X_Test']
    Train_Labels = data['Train_Labels']
    Test_Labels = data['Test_Labels']
    Train_Labels_OH = data['Train_Labels_OH']
    Test_Labels_OH = data['Test_Labels_OH']
    del data

    F_Train = X_Train
    F_Test = X_Test
    del X_Train, X_Test

    F_Test = F_Test.reshape([32,32,3,10000],order='F')
    u1 = np.mean(F_Train, axis=1)
    std1 = np.std(F_Train, axis=1)

    Iter = 10

    Test_Random_Gussian_All_Noise(F_Test, Iter, u1, std1)






