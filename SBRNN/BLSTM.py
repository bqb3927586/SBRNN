import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data
from torch.autograd import Variable

class SBRnn(nn.Module):
    def __init__(self, feature, hidden_size, num_layers):
        super(SBRnn, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.fc2 = nn.Linear(41,10)

    def forward(self, x, h_state):
        #print('x',x)
        r_out, h_state = self.rnn(x, h_state)
        #print('r_out',r_out.size())
        #print('h_state',h_state.size())
        b, s, h = r_out.shape  # (seq, batch, hidden)
        r_out=r_out.contiguous()
        #print(r_out.is_contiguous())
        r_out = r_out.view(-1, h)  # 转化为线性层的输入方式
        #print('r_out', r_out.size())
        out = self.fc(r_out)
        out = out.view(b, s, -1).squeeze()
        out = self.fc2(out)
        #print('out shape:',out.shape,'\n out:\n',out)
        return out

def generate_data(M,Seq_length,train_num):
    train_data=np.random.randint(M,size=[train_num,Seq_length])
    train_label=train_data
    return train_data,train_label

def generate_channel(kop,alpha,beta,t):
    Lambda = kop * (beta ** (-alpha) * t ** (alpha - 1)) * np.exp(-t / beta) / math.gamma(alpha)
    return Lambda

def cal_received_signal(data,feature,Seq_length,omega,num):
    xi=np.zeros([num,Seq_length,feature])
    t=np.arange(0,Seq_length*feature/omega,1/omega)
    t=t.reshape(-1,feature)
    #print(t)
    alpha=2
    kop=1
    beta=0.2
    eta=1
    Lambda=generate_channel(kop,alpha,beta,t*10**6)


    #plt.plot(t,Lambda)
    #plt.grid()
    #plt.show()

    #print(data.shape)
    #print(Lambda.shape)
    #print(data[10,:])
    for step in range(num):
        for j in range(feature):
            xi[step,:,j]=np.convolve(data[step,:],Lambda[:,j])[:Seq_length]+eta

    xi=xi.reshape(num,-1)
    received_signal=np.random.poisson(xi)
    batch, length = received_signal.shape
    diff_receive=received_signal[:,1:]-received_signal[:,:-1]
    b0 = received_signal[:,0].reshape(batch,1)
    bB_1 = received_signal[:,length-1].reshape(batch,1)

    diff_receive=np.concatenate((b0,diff_receive,bB_1),axis=1)
    diff_receive=diff_receive.reshape(batch,-1,1)
    #print('received_signal shape:',received_signal.shape,'signal:\n',received_signal)
    return diff_receive


M = 2  # 4-PAM
train_num=10000
test_num=1000


Seq_length = 10
omega = 20 * 10 ** 6

MAX_EPOCH=50
BATCH_SIZE=50
LR = 0.001

hidden_size=32
num_layers=2

if __name__ == "__main__":
    tao = 0.2 * 10 ** -6 # symbol interval
    a = int(tao * omega) # 过采样率
    model = SBRnn(a,hidden_size,num_layers)

    [data, label] = generate_data(M, Seq_length, train_num)
    received_signal = cal_received_signal(data, a, Seq_length, omega, train_num)
    train_data=torch.from_numpy(received_signal).float()
    train_label=torch.from_numpy(label).float()
    #print(train_data.size())
    #print(train_label.size())
    [data, label] = generate_data(M, Seq_length, test_num)
    received_signal = cal_received_signal(data, a, Seq_length, omega, test_num)
    test_data = torch.from_numpy(received_signal).float()
    test_label = torch.from_numpy(label).float()

    # DataBase in Pytorch
    dataset_train = Data.TensorDataset(train_data,train_label)
    dataset_test = Data.TensorDataset(test_data,test_label)
    train_loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 定义优化器和损失函数
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if 1:
        h_state = None
        for epoch in range(MAX_EPOCH):
            train_loss=0.0
            for step, (x, y) in enumerate(train_loader):
                b_x = Variable(x)
                b_y = Variable(y)
                prediction = model(b_x, h_state)
                loss = loss_func(prediction, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss+=loss
            train_loss=train_loss/test_num*BATCH_SIZE
            #train_loss=train_loss/(train_num*Seq_length)
            print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)


    ber=0
    h_state=None
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        pred = model(b_x, h_state)
        pred = pred.data.numpy().astype(int)
        label = b_y.data.numpy().astype(int)
        no_errors = (pred != label)
        #print(no_errors)
        no_errors = no_errors.astype(int).sum()
        #print(no_errors)
        ber += no_errors / (test_num*Seq_length)
    print(ber)


