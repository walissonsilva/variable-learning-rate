import matplotlib.pyplot as plt  # Para plotar graficos
import numpy as np  # Array do Python
from math  import sqrt, pi

class Backpropagation(object):
    def __init__(self, eta=0.01, epoch_max=10000, Ni=2, Nh=5, Ns=1):
        self.eta = eta
        self.epoch_max = epoch_max
        self.Ni = Ni
        self.Nh = Nh
        self.Ns = Ns
        self.Wini = 0.1

    def load_function(self):
        x = np.arange(-6, 6, 0.2)
        self.N = x.shape[0]
        xmax = np.max(x)

        self.X_train = x / xmax
        self.d = 1 / (1 + np.exp(-1 * x))*(np.cos(x) - np.sin(x))
    
    def train(self):
        self.Wji = np.random.rand(self.Nh, self.Ni) * self.Wini
        self.Wkj = np.random.rand(self.Ns, self.Nh + 1) * self.Wini

        MSE = np.zeros(self.epoch_max)
        plt.ion()

        for epoca in xrange(self.epoch_max):
            deltaWkj = deltaWji = 0
            z = np.zeros(self.N)
            E = np.zeros(self.N)

            for i in xrange(self.N):
                xi = np.array([-1, self.X_train[i]]).reshape(1, -1)
                netj = np.dot(self.Wji, xi.T)
                yj = 1 / (1 + np.exp(-netj.T))
                yj_pol = np.insert(yj[0], 0, -1).reshape(1, -1)
                z[i] = np.dot(self.Wkj, yj_pol.T)[0][0]

                e = self.d[i] - z[i]
                etae = - self.eta * e
                deltaWkj -= np.dot(etae, yj_pol)
                deltaWji -= np.dot(etae * (self.Wkj[:,1:] * yj * (1 - yj)).T, xi)

                E[i] = 0.5 * e**2
            
            self.Wkj += deltaWkj
            self.Wji += deltaWji

            MSE[epoca] = np.sum(E) / self.N

            if (epoca % 200 == 0 or epoca == self.epoch_max - 1):
                if (epoca != 0):
                    plt.cla()
                    plt.clf()
                
                self.plot(z, epoca)
        
        print MSE[-1]
        
        return MSE

    def plot(self, saida, epoca):
        plt.figure(0)
        y, = plt.plot(self.X_train, saida, label="y")
        d, = plt.plot(self.X_train, self.d, '*', label="d")
        plt.legend([y, d], ['Output of Network Neural', 'Desired Value'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Backpropagation')
        plt.text(np.min(self.X_train) - np.max(self.X_train) * 0.17  , np.min(self.d) - np.max(self.d) * 0.17, 'Progress: ' + str(round(float(epoca) / self.epoch_max * 100, 2)) + '%')
        plt.axis([np.min(self.X_train) - np.max(self.X_train) * 0.2, np.max(self.X_train) * 1.2, np.min(self.d) - np.max(self.d) * 0.2, np.max(self.d) * 1.5])
        plt.show()
        plt.pause(1e-10)

    def plot_MSE(self, MSE):
        plt.ioff()
        plt.figure(1)
        plt.title('Mean Square Error (MSE)')
        plt.xlabel('Training Epochs')
        plt.ylabel('MSE')
        plt.plot(np.arange(0, MSE.size), MSE)
        plt.show()

    def show_function(self):
        plt.figure(0)
        plt.title('Second Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(self.X_train, self.d)
        plt.show()

#back = Backpropagation()

#back.load_first_function()
#back.train()