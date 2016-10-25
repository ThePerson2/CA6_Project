from brian2 import *
import math

numberGC = 10

# gT = 0        # target gain. Should vary a bit depending on day of training.
# pT = 0        # target phase shift.
# TauPG = 15*min
w = 10*(1/second)   # rate at which the platform rotates.

MF = NeuronGroup(1,'M = cos(t*w) : 1')
GC = NeuronGroup(numberGC,'''G = cos((t*w) - x) : 1
						x : 1''')
PC = NeuronGroup(1,'P = 1 : 1')
# S = Synapses(GC,PC,pre='P=sum(GC.G[:])')
# Wpg = S.connect(GC,PC)
# print(GC.G[:])
# print(sum(GC.G[:]))
#
#
for i in range(0, numberGC):
 	GC.x[i] = (i/numberGC)*math.pi*2                # makes it so that the delays are distributed evenly
# # help(subplot)
# print(GC.x)
# MF_state = StateMonitor(MF,'M',record=0)
# GC_state = StateMonitor(GC,'G',record=range(0, numberGC))
run(1*second)
print(sum(GC.G[:]))
# figure()
# subplot(211)
# plot(MF_state.t/ms,MF_state.M[0])
#
# for i in range(0, numberGC):
# 	figure()
# 	subplot(211)
# 	plot(GC_state.t/ms,GC_state.G[i])
#
# show()



# El = -65*mV  ## leak potential
# EK = -90*mV  ## potassium equilibrium potential
# ENa = 50*mV  ## sodium equilibrium potential
# tau = 20*ms	 ## time constant
# sigma = 5.*mV
# tau_channel = 2*ms
# tau_m = 0.2*ms
# g = 1000 # g leak is 1, this is relative to that.
# I = 2000*mV
#
# equations =  '''dv/dt = (El-v - g*m**3*h*(v-ENa) - g*n**4*(v-EK) + I)/tau : volt #
#
# 				dm/dt = (1./(1+exp((-v-40*mV)/sigma)) - m)/tau_m : 1
# 				dh/dt = (1./(1+exp( (v+40*mV)/sigma)) - h)/tau_channel : 1
# 				dn/dt = (1./(1+exp((-v-40*mV)/sigma)) - n)/tau_channel : 1'''
#
# neuron = NeuronGroup(1,equations, method='exponential_euler')
# voltage = StateMonitor(neuron,'v',record=0)
# m = StateMonitor(neuron,'m',record=0)
# n = StateMonitor(neuron,'n',record=0)
# h = StateMonitor(neuron,'h',record=0)
# neuron.v = El
# neuron.m = 0
# neuron.h = 1
# neuron.n = 0
#
# run(50*ms)
#
# figure()
# subplot(211)
# plot(voltage.t/ms,voltage.v[0]/mV)
# subplot(212)
# plot(m.t/ms,(m.m[0])**3*h.h[0])
# plot(n.t/ms,(n.n[0])**4)
# show()
