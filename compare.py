import rprop as Rprop
import superSAB as SuperSAB
import backpropagation as Back
import wnn as WNN
import rbf as RBF
import matplotlib.pyplot as plt
import numpy as np

epocas = np.arange(1, 10001, 1)

### BACKPROPAGATION
back = Back.Backpropagation()
back.load_function()
MSE_BP = back.train()

### WNN
wnn = WNN.WNN()
wnn.load_function()
MSE_WNN = wnn.train()

### RBF
rbf = RBF.RBF()
rbf.load_function()
MSE_RBF = rbf.train()

### RPROP
rprop = Rprop.Rprop()
rprop.load_function()
MSE_RP = rprop.train()

### SUPERSAB
super_sab = SuperSAB.SuperSAB()
super_sab.load_function()
MSE_SS = super_sab.train()

plt.ioff()
plt.figure()
RP, = plt.semilogy(epocas, MSE_RP, label="RP")
SS, = plt.semilogy(epocas, MSE_SS, label="SS")
BP, = plt.semilogy(epocas, MSE_BP, label="BP")
WNN, = plt.semilogy(epocas, MSE_WNN, label="WNN")
RBF, = plt.semilogy(epocas, MSE_RBF, label="RBF")
plt.legend([RP, SS, BP, WNN, RBF], ['Rprop', 'SuperSAB', 'Back', 'WNN', 'RBF'])
plt.title('Comparative')
plt.grid(True)
plt.show()