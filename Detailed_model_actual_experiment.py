from brian2 import *
import math

numberGC = 100

gT = 0        # target gain. Should vary a bit depending on day of training.
pT = 0        # target phase shift.
M0=0.25
M1=0.25
G0=1
G1=1
alpha=0.19
alphavm = 0.0000056 # ms -1 

errorDelay = 0
TauPG = 900*second
T = 1.666*(second)   # rate at which the platform rotates.

## make the neuron groups

MF = NeuronGroup(1,'M = M1*cos((((2*math.pi)/T)*t)-(math.pi/2)) + M0 : 1') # Cosine Function for the mossy fibers
GC = NeuronGroup(numberGC,model = '''G =cos((((2*math.pi)/T)*t)-(math.pi/2) - x) + G0 : 1
						x : 1''') # Some phase delays of the above function
for i in range(0, numberGC):
	GC.x[i] = (i/numberGC)*math.pi*2 + alpha*cos((i/numberGC)*math.pi*2)    # makes it so that the delays are distributed evenly


IC = NeuronGroup(1,model='''I : 1''')
PC = NeuronGroup(1,model = '''E : 1
							I : 1
							Eini : 1
							Iini : 1
							P=E-I : 1
							Pini=Eini-Iini : 1
							V : 1''')  # The functions are defined by the synapses later. P is the purkinje cell activity, V is MVN activity, which is sent back so that it cen be used for the weight calculation.
MVN = NeuronGroup(1,model = '''	M : 1
								P : 1
								Pini : 1
								Wvm : 1
								Ve0 = 2.25 : 1
								VE = 2*Wvm*(M-M0)-P+Ve0 : 1
								VI=M : 1
								V = VE-VI : 1
								eyeMovement = -V : 1''') # eyeMovement is the only new variable. it is the

## make the synapses

# Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
# 								Wpg = x_pre : 1''')
#Sig stands for interneuron-granularcells
Sig=Synapses(GC,IC,model='''I_post=(Wig/numberGC)*G_pre-I0/numberGC : 1 (summed)
							Iini_post=(Wigini/numberGC)*G_pre-I0/numberGC : 1 (summed)
								Wig = 2.5 : 1
								Wigini = 2.5 : 1
								I0 = Wig*G0 - 0.85''')
#need to figure out how (summed) works with constants !!!

Sip=Synapses(IC,PC,model='''I_post=Wpi*I_pre : 1
							Wpi=1''')

#Spg stands for granular-Purkinje
Spg = Synapses(GC,PC,model='''E_post = (1/numberGC)*Wpg*G_pre : 1 (summed)
								Eini_post = (1/numberGC)*Wpgini*G_pre : 1 (summed)
								dWpg/dt = ((V_post - gT*cos(w*t + pT))*G_pre)/TauPG : 1/second
								Wpgini = 1.85''')
 # sums granule cell activity into the purkinje cell, as weights are applied.
#Smv stands for mossy-MVN
Smv = Synapses(MF,MVN, model='''M_post = M_pre : 1 (summed)
								dWvm_post/dt=alphavm*(M0-M_pre)(P_post-''')  ## I made them summed because that makes it work.
#Spv stands for Purkinje-MVN
Spv = Synapses(PC,MVN, model='''P_post = P_pre : 1 (summed)
								V_pre = V_post : 1 (summed)''')  ## I made them summed because that makes it work.

## connect the synapses


Sig.connect()
Sip.connect()
Spg.connect()
Smv.connect()
Spv.connect()

## create the state monitors

# You can flag these on or off to see the graphs they produce.
MF_state = StateMonitor(MF,'M',record=0)
GC_state = StateMonitor(GC,'G',record=True)
Weight_state = StateMonitor(Spg,'Wpg',record=True)
PC_state = StateMonitor(PC,'P',record=0)
MVN_state = StateMonitor(MVN,'eyeMovement',record=0)

## run the model

# They are set up like this to immitate the paradigm for the paper. 
run(5.0*second)

gT = -0.5

run(5.0*second)

gT = -1

run(5.0*second)
run(5.0*second)

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
