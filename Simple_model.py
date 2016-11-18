from brian2 import *
import math

numberGC = 10

gT = 1        # target gain. Should vary a bit depending on day of training.
pT = 0        # target phase shift.
errorDelay = 0
TauPG = 1*second
w = math.pi*3*(1/second)   # rate at which the platform rotates.

## make the neuron groups

MF = NeuronGroup(1,'M = cos(t*w) : 1') # upon some reading... I'm not sure that the cosine was a valid choice. Will look into this more.
GC = NeuronGroup(numberGC,model = '''G = cos((t*w) + x) : 1
						x : 1''')
for i in range(0, numberGC):
 	GC.x[i] = (i/numberGC)*math.pi*2        # makes it so that the delays are distributed evenly
PC = NeuronGroup(1,model = '''P : 1
							  V : 1''')  # V from MVN neuron
MVN = NeuronGroup(1,model = '''M : 1
							   P : 1
							   V = M - P : 1
							   eyeMovement = -V : 1''')

## make the synapses

# Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
# 								Wpg = x_pre : 1''')
Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
								dWpg/dt = ((V_post - gT*cos(w*t + pT))*G_pre)/TauPG : 1/second''')
Smv = Synapses(MF,MVN, model='''M_post = M_pre : 1 (summed)''')  ## I made them summed because that makes it work.
Spv = Synapses(PC,MVN, model='''P_post = P_pre : 1 (summed)
								V_pre = V_post : 1 (summed)''')  ## I made them summed because that makes it work.

## connect the synapses

Spg.connect()
Smv.connect()
Spv.connect()

## create the state monitors

MF_state = StateMonitor(MF,'M',record=0)
# GC_state = StateMonitor(GC,'G',record=True)
#Weight_state = StateMonitor(Spg,'Wpg',record=True)
PC_state = StateMonitor(PC,'P',record=0)
MVN_state = StateMonitor(MVN,'eyeMovement',record=0)

## run the model

# run(0.1*second)
# print(MVN.M[:])
# print(Smv.M_post[:])
# print(Smv.M_pre[:])
# print()
# run(0.1*second)
# print(MVN.M[:])
# print(Smv.M_post[:])
# print(Smv.M_pre[:])
run(10*second)

## plot results

if 'MF_state' in locals():
	figure()
	subplot(211)
	plot(MF_state.t/ms,MF_state.M[0])

if 'GC_state' in locals():
	for i in range(0, numberGC):
		figure()
		subplot(211)
		plot(GC_state.t/ms,GC_state.G[i])
if 'Weight_state' in locals():
	for i in range(0, numberGC):
		figure()
		# subplot(211)
		plot(Weight_state.t/ms,Weight_state.Wpg[i])
if 'PC_state' in locals():
	figure()
	subplot(211)
	plot(PC_state.t/ms,PC_state.P[0])

if 'MVN_state' in locals():
	figure()
	subplot(211)
	plot(MVN_state.t/ms,MVN_state.eyeMovement[0])
	xlabel('time')
	ylabel('Eye movement')

show()
