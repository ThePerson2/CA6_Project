from brian2 import *
import math
import queue

numberGC = 10

defaultclock.dt = 0.01*second
gT = 0        # target gain. Should vary a bit depexnding on day of training.
pT = 0        # target phase shift.
unitlessErrorDelay = 0 # set the delay here so that the file prints right
errorDelay =  unitlessErrorDelay*second
if errorDelay != 0:
	delayQueue = queue.Queue(maxsize = int(errorDelay/defaultclock.dt))
	startMVN = 1
	for i in range(0,int(errorDelay/defaultclock.dt)):
		delayQueue.put(startMVN) # fill the queue with initial values
	Vdelayed = delayQueue.get() # get one of them, open 
TauPG = 15*60*second
w = 0.6*(1/second)   # rate at which the platform rotates.
vCF = 0
FiftyMinutesInTimeSteps = int(((second)/defaultclock.dt)*60*50)

## make the neuron groups

MF = NeuronGroup(1,'M = cos(t*w) : 1') # Cosine Function for the mossy fibers
GC = NeuronGroup(numberGC,model = '''G = cos((t*w) + x) : 1
						x : 1''') # Some phase delays of the above function
for i in range(0, numberGC):
 	GC.x[i] = (i/numberGC)*math.pi*2        # makes it so that the delays are distributed evenly
PC = NeuronGroup(1,model = '''P : 1
							  V : 1''')  # The functions are defined by the synapses later. P is the purkinje cell activity, V is MVN activity, which is sent back so that it cen be used for the weight calculation.
MVN = NeuronGroup(1,model = '''M : 1
							   P : 1
							   V = M - P : 1''')

## make the synapses

# Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
# 								Wpg = x_pre : 1''')
if (errorDelay == 0):
	Spg = Synapses(GC,PC,method='euler',model='''P_post = Wpg*G_pre : 1 (summed)
								dWpg/dt = ((V_post - gT*cos(w*(t + pT)))*G_pre)/TauPG : 1/second''') # sums granule cell activity into the purkinje cell, as weights are applied.
else:
	Spg = Synapses(GC,PC,method='euler',model='''P_post = Wpg*G_pre : 1 (summed)
								dWpg/dt = ((Vdelayed - gT*cos(w*(t - errorDelay + pT)))*G_pre)/TauPG : 1/second''') # sums granule cell activity into the purkinje cell, as weights are applied. # Vpast[int(t/(defaultclock.dt*second))] - 

Smv = Synapses(MF,MVN, model='''M_post = M_pre : 1 (summed)''')  ## I made them summed because that makes it work.
Spv = Synapses(PC,MVN, model='''P_post = P_pre : 1 (summed)
								V_pre = V_post : 1 (summed)''')  ## I made them summed because that makes it work.

## connect the synapses

Spg.connect()
Smv.connect()
Spv.connect()

## create the state monitors

# You can flag these on or off to see the graphs they produce.
MF_state = StateMonitor(MF,'M',record=0)
#GC_state = StateMonitor(GC,'G',record=True)
Weight_state = StateMonitor(Spg,'Wpg',record=True)
PC_state = StateMonitor(PC,'P',record=0)
MVN_state = StateMonitor(MVN,'V',record=0)

## run the model

# They are set up like this to immitate the paradigm for the paper.
if errorDelay != 0:
	for i in range(0,FiftyMinutesInTimeSteps):
		run(defaultclock.dt)
		#print(MVN_state.V[0])
		#print(list(MVN_state.V[0])[-1])
		Vpast = delayQueue.put(list(MVN_state.V[0])[-1])
		Vdelayed = delayQueue.get()
		#print(Vdelayed)
	gT = -0.5
	print("report")
	for i in range(0,FiftyMinutesInTimeSteps):
		run(defaultclock.dt)
		Vpast = delayQueue.put(list(MVN_state.V[0])[-1])
		Vdelayed = delayQueue.get()
	gT = -1
	print("report")
	for i in range(0,FiftyMinutesInTimeSteps*2):
		run(defaultclock.dt)
		Vpast = delayQueue.put(list(MVN_state.V[0])[-1])
		Vdelayed = delayQueue.get()
		
else:
	run(FiftyMinutesInTimeSteps*second)
	gT = -0.5
	run(FiftyMinutesInTimeSteps*second)
	gT = -1
	run(FiftyMinutesInTimeSteps*2*second)

# gT = -0.5
#
# for i in range(0,5):
# 	run(0.1*second)
# gT = -1
#
# for i in range(0,10):
# 	run(0.1*second)

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
	plot(MVN_state.t/ms,-MVN_state.V[0])
	xlabel('time')
	ylabel('Eye movement')

show()
numpy.savetxt("MVN_state_" + str(unitlessErrorDelay) + ".csv", MVN_state.V[0], delimiter=",")
numpy.savetxt("MF_state_" + str(unitlessErrorDelay) + ".csv", MF_state.M[0], delimiter=",")




