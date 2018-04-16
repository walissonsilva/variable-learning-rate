import rprop as Rprop
import superSAB as SuperSAB

rprop = Rprop.Rprop()
rprop.load_function()
MSE = rprop.train()

rprop.plot_MSE(MSE)

super_sab = SuperSAB.SuperSAB()

super_sab.load_function()
MSE = super_sab.train()
super_sab.plot_MSE(MSE)