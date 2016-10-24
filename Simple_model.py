from brian2 import *

gT = 0        # target gain. Should vary a bit depending on day of training.
pT = 0        # target phase shift.



equations = '''MF = cos(t) : 1           # Mossy fiber response
			   GC = cos(t - x)          # Granule Cell response. There are "N" granule cells, and "x" is the unique phase of each cell.
			   PC = sum(Wpg*G)          # so the "sum" business is "for each x". The way I wrote it will not work. Wpg is a function of x too
			   MVN = MF - PC            # each as a function of time.
			   MVN_target =
			   '''


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
