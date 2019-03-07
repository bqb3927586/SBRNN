import numpy as np
import matplotlib.pyplot as plt


def cal_received_signal(data, kmo, c, mu, tao, omega, eta):
    a = int(omega * tao)
    Seq_length = len(data)
    epos=0.0000001
    t = np.arange(epos, Seq_length * tao + epos, 1 / omega)
    t = t.reshape(-1, a)
    print(t)
    xi=np.zeros([Seq_length,a])
    Lambda = kmo * np.sqrt(c / (2 * np.pi * t ** 3)) * np.exp(-(c * (t - mu) ** 2) / (2 * mu ** 2 * t))


    #plt.plot(t,Lambda)
    #plt.grid()
    #plt.show()

    #print(data.shape)
    #print(Lambda.shape)
    #print(data[10,:])
    for j in range(a):
        #print('data:',data)
        #print('Lambda[',j,']:',Lambda[:,j])
        #print('result:',np.convolve(data[:],Lambda[:,j]))
        xi[:,j]=np.convolve(data[:],Lambda[:,j])[:Seq_length]+eta
        print('xi:[',j,']:',xi[:,j])

    #print(xi[10,:,:])
    received_signal=np.random.poisson(xi)

    t = t.reshape(a * Seq_length, -1)
    received_signal = received_signal.reshape(a * Seq_length, -1)
    xi = xi.reshape(a * Seq_length, -1)
    plt.plot(t,received_signal)
    plt.grid()
    plt.show()


kmo=100
c=8
mu=40
tao=1
omega=100
eta=1

data=np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,0],dtype=float)

a=int(omega*tao)
cal_received_signal(data,kmo,c,mu,tao,omega,eta)
print(data)