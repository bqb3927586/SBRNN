import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data
from torch.autograd import Variable

class SBRnn(nn.Module):
    def __init__(self, feature, hidden_size, num_layers, M):
        super(SBRnn, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = M
        self.rnn = nn.LSTM(
            input_size = feature,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, self.out_size),nn.Softmax(dim=1))

    def forward(self, x, L):
        #print('x',x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda() # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()
        batch, Seq_length, samples = x.size()
        sliding_num = Seq_length - L +1
        sliding_res = torch.zeros([batch, Seq_length, self.out_size]).cuda()
        jk = torch.zeros(Seq_length).cuda()
        for i in range(len(jk)):
            if i<L-1:
                jk[i]=i+1
            elif i>len(jk)-L:
                jk[i]=len(jk)-i
            else:
                jk[i]=L

        for count in range(sliding_num):
            start = count
            end = count + L
            sliding_x = x[:,start:end,:]
            sliding_x, _ = self.rnn(sliding_x, (h0,c0)) #out.shape=[batch,L,hidden_size]
            _, _, hidden_size = sliding_x.shape
            sliding_x = sliding_x.contiguous()
            sliding_x = sliding_x.view(-1,hidden_size)
            sliding_x = self.fc(sliding_x)
            sliding_x = sliding_x.view(batch, -1, self.out_size)
            sliding_res[:,start:end,:] = sliding_res[:,start:end,:] + sliding_x

        for i in range(len(jk)):
            sliding_res[:,i,:] =  sliding_res[:,i,:].clone()/jk[i]
        #sliding_res.backward(torch.ones(size=sliding_res.size()))
        return sliding_res

def generate_data(M,Seq_length,data_num):
    origin_data=np.random.randint(M,size=[data_num,Seq_length])
    return origin_data

def cal_received_signal(data,samples,Seq_length,omega,B,num):
    t = np.arange(0,Seq_length*samples/omega-10**-10,1/omega)
    t = t.reshape(-1,samples)
    t = t*10**6
    #print(t)
    alpha = 2
    kop = 10
    beta = 0.2
    eta = 1

    Lambda = kop * (beta ** (-alpha) * t ** (alpha - 1)) * np.exp(-t / beta) / math.gamma(alpha)
    '''
    plt.plot(t,Lambda)
    plt.grid()
    plt.show()
    '''
    #print(data.shape)
    #print(Lambda.shape)
    #print(data[10,:])
    xi = np.zeros([num,Seq_length,samples])
    for step in range(num):
        for j in range(samples):
            xi[step,:,j]=np.convolve(data[step,:],Lambda[:,j])[:Seq_length]+eta

    received_signal = np.random.poisson(xi) / B
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num, Seq_length, int(samples/B), B)
    #print(received_signal[0,:,:,:])
    received_signal = np.sum(received_signal,axis=3).squeeze()
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num,-1)
    #print(received_signal[0,:])
    batch, length = received_signal.shape
    diff_receive=received_signal[:,1:]-received_signal[:,:-1]
    b0 = received_signal[:,0].reshape(batch,1)
    bB_1 = received_signal[:,length-1].reshape(batch,1)
    diff_receive=np.concatenate((b0,diff_receive),axis=1)
    diff_receive=diff_receive.reshape(batch,Seq_length,-1)
    #print('received_signal shape:',received_signal.shape,'signal:\n',received_signal)
    return diff_receive


M = 2  # 4-PAM
train_num = 200*10**3
test_num = 1000

L = 50
Seq_length = 100
omega = 2 * 10 ** 9

MAX_EPOCH = 20
BATCH_SIZE = 128
LR = 0.001

hidden_size = 80
num_layers = 3

if __name__ == "__main__":
    tao = 0.05 * 10 ** -6 # symbol interval
    a = int(tao * omega) # 过采样率
    B = 10
    model = SBRnn(int(a/B),hidden_size,num_layers,M)
    model.cuda()
    print(model)
    label = generate_data(M, Seq_length, train_num)
    received_signal = cal_received_signal(label, a, Seq_length, omega, B, train_num)

    train_data=torch.from_numpy(received_signal).float()
    train_label=torch.from_numpy(label).long()

    #print(train_data.size())
    #print(train_label.size())

    label = generate_data(M, Seq_length, test_num)
    received_signal = cal_received_signal(label, a, Seq_length, omega, B, test_num)

    test_data = torch.from_numpy(received_signal).float()
    test_label = torch.from_numpy(label).long()

    # DataBase in Pytorch
    dataset_train = Data.TensorDataset(train_data,train_label)
    dataset_test = Data.TensorDataset(test_data,test_label)
    train_loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 定义优化器和损失函数
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if 1:
        print('train begin\n')
        for epoch in range(MAX_EPOCH):
            train_loss=0.0
            for step, (x, y) in enumerate(train_loader):
                b_x = x.requires_grad_().cuda()
                b_y = y.cuda()
                optimizer.zero_grad()
                prediction = model(b_x, L)
                prediction = prediction.reshape(-1,M)
                b_y = b_y.reshape(-1)
                loss = loss_func(prediction, b_y)
                loss.backward()
                optimizer.step()
                train_loss+=loss
            train_loss=train_loss/test_num
            print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)


    ber=0
    print('test begin\n')
    for step, (x, y) in enumerate(test_loader):
        pred = model(x.cuda(), L)
        pred = pred.cpu()
        pred = pred.detach().numpy().astype(int)
        pred = np.argmax(pred, axis=2)
        label = y.detach().numpy()
        no_errors = (pred != label)
        no_errors = no_errors.astype(int).sum()
        #print(no_errors)
        ber += no_errors / (test_num*Seq_length)
    print(ber)


