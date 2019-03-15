import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data
from torch.autograd import Variable
import time

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
        batch, Seq_Length, samples = x.size()
        sliding_num = Seq_Length - L +1
        sliding_res = torch.zeros([batch, Seq_Length, self.out_size]).cuda()
        jk = torch.zeros(Seq_Length).cuda()
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

def generate_data(M,Seq_Length,data_num):
    origin_data=np.random.randint(M,size=[data_num,Seq_Length])
    return origin_data

def cal_received_training_signal(data,samples,Seq_Length,omega,B,num):
    t = np.arange(0,Seq_Length*samples/omega-10**-10,1/omega)
    t = t.reshape(-1,samples)
    t = t*10**6
    #print(t)
    alpha = 2
    kop = 10
    beta = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
    eta = [1, 10, 20, 50, 100, 200, 500]
    Lambda = np.zeros(shape=[Seq_Length,samples])
    xi = np.zeros([num, Seq_Length, samples])

    for step in range(num):
        beta_index = np.random.randint(0, len(beta))
        eta_index = np.random.randint(0, len(eta))
        Lambda = kop * (beta[beta_index] ** (-alpha) * t ** (alpha - 1)) * np.exp(-t / beta[beta_index]) / math.gamma(alpha)
        for j in range(samples):
            xi[step,:,j] = np.convolve(data[step,:],Lambda[:,j])[:Seq_Length] + eta[eta_index]

    received_signal = np.random.poisson(xi) / B
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num, Seq_Length, int(samples/B), B)
    #print(received_signal[0,:,:,:])
    received_signal = np.sum(received_signal,axis=3).squeeze()
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num,-1)
    #print(received_signal[0,:])
    batch, length = received_signal.shape
    b_avg = np.mean(received_signal,axis=1)
    b_sigma = np.sqrt(np.var(received_signal,axis=1))
    for i in range(batch):
        received_signal[i,:] = (received_signal[i,:] - b_avg[i]) / b_sigma[i]
    diff_receive=received_signal[:,1:]-received_signal[:,:-1]
    b0 = received_signal[:,0].reshape(batch,1)
    bB_1 = received_signal[:,length-1].reshape(batch,1)

    diff_receive=np.concatenate((b0,diff_receive),axis=1)
    diff_receive=diff_receive.reshape(batch,Seq_Length,-1)
    #print('diff_receive shape:', diff_receive.shape,'signal:\n',diff_receive)
    return diff_receive

def cal_received_testing_signal(data,samples,Seq_Length,omega,B,num):
    t = np.arange(0,Seq_Length*samples/omega-10**-10,1/omega)
    t = t.reshape(-1,samples)
    t = t*10**6
    #print(t)
    alpha = 2
    kop = 10
    beta_0 = 0.2
    eta_0 = 10
    v = 10 ** -3
    d = 10 ** -3
    beta = beta_0
    eta = eta_0
    Lambda = np.zeros(shape=[Seq_Length,samples])

    for i in range(Seq_Length):
        Lambda[i,:] = kop * (beta ** (-alpha) * t[i,:] ** (alpha - 1)) * np.exp(-t[i,:] / beta) / math.gamma(alpha)
        N =np.random.normal()
        beta = beta + d * beta_0 * N + v * beta_0
        eta = eta + d * eta_0 * N + v * eta_0

    '''
    plt.plot(t,Lambda)
    plt.grid()
    plt.show()
    '''
    #print(data.shape)
    #print(Lambda.shape)
    #print(data[10,:])
    xi = np.zeros([num,Seq_Length,samples])
    for step in range(num):
        for j in range(samples):
            xi[step,:,j]=np.convolve(data[step,:],Lambda[:,j])[:Seq_Length]+eta

    received_signal = np.random.poisson(xi) / B
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num, Seq_Length, int(samples/B), B)
    #print(received_signal[0,:,:,:])
    received_signal = np.sum(received_signal,axis=3).squeeze()
    #print(received_signal[0,:,:])
    received_signal = received_signal.reshape(num,-1)
    #print(received_signal[0,:])
    batch, length = received_signal.shape
    b_avg = np.mean(received_signal,axis=1)
    b_sigma = np.sqrt(np.var(received_signal,axis=1))
    for i in range(batch):
        #received_signal[i,:] = (received_signal[i,:] - b_avg[i]) / b_sigma[i]
        received_signal[i, :] = (received_signal[i, :] - b_avg[i])
    diff_receive=received_signal[:,1:]-received_signal[:,:-1]
    b0 = received_signal[:,0].reshape(batch,1)
    bB_1 = received_signal[:,length-1].reshape(batch,1)
    diff_receive=np.concatenate((b0,diff_receive),axis=1)
    diff_receive=diff_receive.reshape(batch,Seq_Length,-1)
    #print('received_signal shape:',received_signal.shape,'signal:\n',received_signal)
    return diff_receive

def train_model(model,L,M,Seq_Length,train_loader,train_num,test_loader,test_num,start_epoch,MAX_EPOCH,loss_min,loss_func,optimizer):
    print('train begin\n')
    ber_record = np.zeros(MAX_EPOCH)
    train_loss = np.zeros(MAX_EPOCH)
    for epoch in range(start_epoch,MAX_EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.requires_grad_().cuda()
            b_y = y.cuda()
            optimizer.zero_grad()
            prediction = model(b_x, L)
            prediction = prediction.reshape(-1, M)
            b_y = b_y.reshape(-1)
            loss = loss_func(prediction, b_y)
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss
        train_loss[epoch] = train_loss[epoch] / (train_num)
        if train_loss[epoch] < loss_min:
            loss_min = train_loss[epoch]
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'loss': loss_min,'optimizer': optimizer.state_dict(), },
                       PATH + '/m-' + timestampLaunch + '.pth.tar')
            print('Epoch: ', epoch + 1, ' | [save] | train loss: %.4f' % loss_min)
        else:
            print('Epoch: ', epoch + 1, ' | [----] | train loss: %.4f' % loss_min)
        ber_record[epoch]=test_model(model,L,Seq_Length,test_loader,test_num)
    '''
    plt.plot(np.arange(0,MAX_EPOCH),train_loss)
    plt.plot(np.arange(0,MAX_EPOCH),ber_record)
    plt.grid()
    plt.show()
    '''
    if epoch == MAX_EPOCH - 1:
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'loss': loss_min,'optimizer': optimizer.state_dict(),},
                    PATH + '/m-fianl-' + timestampLaunch + '.pth.tar')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(0,MAX_EPOCH)+1,train_loss)
    ax1.set_ylabel('Train Loss')
    ax1.set_title("Train Loss & BER")
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.arange(0,MAX_EPOCH)+1,ber_record, 'r')
    #ax2.set_xlim([0, np.e])
    ax2.set_ylabel('Test BER')
    ax2.set_xlabel('Epoch')
    plt.savefig('result2.png')
    plt.show()


def test_model(model,L,Seq_Length,test_loader,test_num):
    ber=0
    #print('test begin\n')
    for step, (x, y) in enumerate(test_loader):
        pred = model(x.cuda(), L)
        pred = pred.cpu()
        pred = pred.detach().numpy().astype(int)
        pred = np.argmax(pred, axis=2)
        label = y.detach().numpy()
        no_errors = (pred != label)
        no_errors = no_errors.astype(int).sum()
        #print(no_errors)
        ber += no_errors / (test_num * Seq_Length)
    print('BER:',ber)
    return ber

M = 2  # OOK
train_num = 200*10**2 #200K train Seq
test_num = 1000 # 1K test Seq

L = 50 # Sliding window Length
Seq_Length = 100 # total Seq length
omega = 2 * 10 ** 9 # sampling rate

MAX_EPOCH = 50
BATCH_SIZE = 128
LR = 0.001

hidden_size = 80
num_layers = 3

gen_flag = 0
load_flag = 0
train_flag = 1
test_flag = 0
PATH='./model'
load_path='./model/m-12032019-130125.pth.tar'
if __name__ == "__main__":
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    tao = 0.05 * 10 ** -6 # symbol interval
    a = int(tao * omega) # 过采样率
    B = 10 # 降采样率

    model = SBRnn(int(a/B),hidden_size,num_layers,M).cuda()
    print(model)

    if gen_flag:
        train_label = generate_data(M, Seq_Length, train_num)
        train_data = cal_received_training_signal(train_label, a, Seq_Length, omega, B, train_num)
        #np.save("train_data.npy", train_data)
        #np.save("train_label.npy", train_label)
        test_label = generate_data(M, Seq_Length, test_num)
        test_data = cal_received_testing_signal(test_label, a, Seq_Length, omega, B, test_num)
        #np.save("test_data.npy", test_data)
        #np.save("test_label.npy", test_label)
    else:
        train_data = np.load("train_data.npy")
        train_label = np.load("train_label.npy")
        test_data = np.load("test_data.npy")
        test_label = np.load("test_label.npy")

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()

    # DataBase in Pytorch
    dataset_train = Data.TensorDataset(train_data,train_label)
    dataset_test = Data.TensorDataset(test_data,test_label)
    train_loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 定义优化器和损失函数
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    loss_min = 10000
    if load_flag:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss_min = checkpoint['loss']
        print('Epoch: ', start_epoch, '| train loss: %.4f' % loss_min)
        #model.eval()

    if train_flag:
        train_model(model,L,M,Seq_Length,train_loader,train_num,test_loader,test_num,start_epoch,MAX_EPOCH,loss_min,loss_func,optimizer)

    if test_flag:
        test_model(model, L, Seq_Length, test_loader, test_num)