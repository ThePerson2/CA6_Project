from brian2 import *
import math
import numpy as np

numberGC = 100
defaultclock.dt = 0.001*second
runFullExperiment = 1 # set to 1 if you want to run the whole thing. 0 to screw around elsewhere. 
gT = 0        # target gain. Should vary a bit depending on day of training.
pT = 0        # target phase shift.
M0=0.25
M1=0.25
G0=1 # Set to 1.8 for excite mutants
G1=1
Ve0 = 2.25
Vt0=1
H=0.03
L=1
alpha=0.19
alphavm = 0.0000056 # ms -1 
alphaD=4.5*(1e-6)*1e3*(1/second)
alphaPG=3.5*(1e-5)*1e3*(1/second)
alphaVM=5.5*(1e-6)*1e3*(1/second)
pie=math.pi
sigma=0.02
errorDelay = 0*ms
TauPG = 900*second
T = 1.666*(second)   # rate at which the platform rotates.
trialInt = 50*T
nightInt = 1440*T

gTphase = pie/2 # equations 20 and 21 differ only in this term. When gT goes negative, I will change it through this variable. 

Wig = 2.5
Wpi = 1
Wigini = 2.5
I0 = Wig*G0 - 0.85
Wpgini = 1.85
gibbi=0

eqM='M = M1*cos((((2*math.pie)/T)*t)-(math.pie/2)) + M0 : 1'

eqG='G =cos((((2*math.pie)/T)*t)-(math.pie/2) - x) + G0 : 1'+'    x:1'
eqX='GC.x[i]=(i/numberGC)*math.pie*2 + alpha*cos((i/numberGC)*math.pie*2)'

eqI='I_post=(Wig/numberGC)*Gtot_post-I0 : 1'
eqI_ini='Iini_post=(Wigini/numberGC)*Gtot_post-I0: 1'
eqGtot='Gtot_post=G_pre : 1 (summed)'

eqP1='P1=(1/numberGC)*Wpg*G_pre : 1 (summed)'
eqP2='P2=Wpi*I_pre'
eqWpg='dWpg/dt=((V_post - gT*cos(w*t + pT))*G_pre)/TauPG : 1/second'

eqA='A=2*Wvm*(M_pre-M0)'
eqP='P_post=P_pre'


## make the neuron groups

MF = NeuronGroup(1,'''M = M1*cos((((2*pie)/T)*t)-(pie/2))+M0 : 1''') 
# Cosine Function for the mossy fibers

GC = NeuronGroup(numberGC,
	'''G = G1*cos((((2*pie)/T)*t)-(pie/2)-x) + G0 : 1
	x : 1
	''')
# Some phase delays of the above function
	
for i in range(0, numberGC):
	GC.x[i] = (i/numberGC)*pie*2 + alpha*cos((i/numberGC)*pie*2)
# makes it so that the delays are distributed evenly


IC = NeuronGroup(1,
	'''
	Gtot : 1
	I = (Wig/numberGC)*Gtot-I0: 1
	Iini = (Wigini/numberGC)*Gtot-I0 : 1
	''') # Iini = I, basically always no matter what, and it is supposed to. 
						
PC = NeuronGroup(1,
	'''
	P1 : 1
	P2 : 1
	P1ini : 1
	P2ini : 1
	P=(P1/numberGC)-P2 : 1
	Pini=(P1ini/numberGC)-P2ini : 1
	C : 1
	V : 1
	''')  
	
# The functions are defined by the synapses later. P is the purkinje cell activity, V is MVN activity, which is sent back so that it cen be used for the weight calculation.

MVN = NeuronGroup(1, 
	''' 
	P : 1
	Pini : 1
	A : 1
	VE = A-P+Ve0 : 1
	VI : 1
	V = VE-VI : 1
	eyeMovement = -V : 1''') # eyeMovement is the only new variable. it is the
							
CF=NeuronGroup(1,
	'''
	V : 1
	Vt : 1
	M : 1
	C=-L*(V-Vt)-H*(M-M0) : 1
	''')

VT=NeuronGroup(1,
	'''
	Vt=abs(gT)*M1*cos((2*pie*t/T)-(gTphase))+Vt0 : 1
''')

## make the synapses

# Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
# 								Wpg = x_pre : 1''')

Sig=Synapses(GC,IC,
	'''
	Gtot_post=G_pre : 1 (summed)
	''')

Sip=Synapses(IC,PC,
	'''
	P2_post=Wpi*I_pre : 1 (summed)
	P2ini_post=Wpi*Iini_pre : 1 (summed)
	''')

Spg = Synapses(GC,PC,
	'''
	P1_post = Wpg*G_pre : 1 (summed)
	P1ini_post = Wpgini*G_pre : 1 (summed)
	dWpg/dt = (-alphaPG*C_post+sqrt(alphaPG)*sigma*(randn()))*G_pre+alphaD*(Wpgini-Wpg) : 1 (clock-driven)
	''')
Spg.connect()
Spg.Wpg=1.85
# sums granule cell activity into the purkinje cell, as weights are applied.

Smv = Synapses(MF,MVN,
	'''
	VI_post=M_pre : 1 (summed)
	A_post=2*Wvm*(M_pre-M0) : 1 (summed)
	dWvm/dt=alphavm*(M0-M_pre)*(P_post-Pini_post) : 1 (clock-driven)
	''')
Smv.connect()
Smv.Wvm=0.88  # make sure Wvm changes. # Set to 1.19 for inhibitory mutants. # Set to 0.7 for excitatory mutants. 

Spv = Synapses(PC,MVN,
	'''
	P_post = P_pre : 1 (summed)
	Pini_post=Pini_pre : 1 (summed)
	''')

Svc=Synapses(MVN,CF,
	'''
	V_post=V_pre : 1 (summed)
	''')

Smc=Synapses(MF,CF,
	'''
	M_post=M_pre : 1 (summed)
	''')
	
Stc=Synapses(VT,CF,
	'''
	Vt_post=Vt_pre : 1 (summed)
	''')
	
Scp=Synapses(CF,PC,
	'''
	C_post=C_pre : 1 (summed)
	''')
	

#need initialize Wvm,Wpg



## connect the synapses


Sig.connect()
Sip.connect()
#Spg.connect()
#Smv.connect()
Spv.connect()
Svc.connect()
Smc.connect()
Stc.connect()
Scp.connect()

## create the state monitors

# You can flag these on or off to see the graphs they produce.
#MF_state = StateMonitor(MF,'M',record=0)
#GC_state = StateMonitor(GC,'G',record=True)
Wpg_state = StateMonitor(Spg,'Wpg',record=True)
Wvm_state = StateMonitor(Smv,'Wvm',record=True)
PC_state = StateMonitor(PC,'P',record=0)
Investigate = StateMonitor(IC,'I',record=0)# set the investigates to whatever you want to track for a given trial.
Investigate2 = StateMonitor(VT,'Vt',record=0) 
Investigate3 = StateMonitor(CF,'C',record=0) 
Investigate4 = StateMonitor(MVN,'A',record=0)
MVN_state = StateMonitor(MVN,'eyeMovement',record=0)

## run the model

# They are set up like this to immitate the paradigm for the paper. 
if runFullExperiment:
	gT = 1
	L = 1
	run(trialInt)
	L = 0
	run(nightInt*2)
	gT = 0 
	L = 1
	run(trialInt)
	L = 0
	run(nightInt)
	gT = -0.5
	gTphase = -(pie/2)
	L = 1
	run(trialInt)
	L = 0
	run(nightInt)
	gT = -1 
	L = 1
	run(trialInt)
	L = -1
	run(nightInt)
#gT = -0.5

#run(5.0*second)

#gT = -1

#run(5.0*second)
#run(5.0*second)

## plot results


if 'MF_state' in locals():
	figure()
	subplot(211)
	plot(MF_state.t/ms,MF_state.M[0])
	xlabel('time')
	ylabel('MF')
	
if 'GC_state' in locals():
	for i in range(0, int(numberGC/10)):
		figure()
		subplot(211)
		plot(GC_state.t/ms,GC_state.G[i*10])
		xlabel('time')
		ylabel('GC')
if 'Wpg_state' in locals():
	for i in range(0, int(numberGC/10)):
		figure()
		# subplot(211)
		plot(Wpg_state.t/ms,Wpg_state.Wpg[i*10])
if 'Wvm_state' in locals():
	figure()
	subplot(211)
	plot(Wvm_state.t/ms,Wvm_state.Wvm[0])
	xlabel('time')
	ylabel('Wvm')
if 'PC_state' in locals():
	figure()
	subplot(211)
	plot(PC_state.t/ms,PC_state.P[0])
	xlabel('time')
	ylabel('PC')
if 'Investigate' in locals():
	for i in range(0, 1):
		figure()
		# subplot(211)
		plot(Investigate.t/ms,Investigate.I[i])
	xlabel('time')
	ylabel('Investigate')
if 'Investigate2' in locals():
	for i in range(0, 1):
		figure()
		# subplot(211)
		plot(Investigate2.t/ms,Investigate2.Vt[i])
	xlabel('time')
	ylabel('Investigate2')
if 'Investigate3' in locals():
	for i in range(0, 1):
		figure()
		# subplot(211)
		plot(Investigate3.t/ms,Investigate3.C[i])
	xlabel('time')
	ylabel('Investigate3')
if 'Investigate4' in locals():
	for i in range(0, 1):
		figure()
		# subplot(211)
		plot(Investigate4.t/ms,Investigate4.A[i])
	xlabel('time')
	ylabel('Investigate4')
	
if 'MVN_state' in locals():
	figure()
	subplot(211)
	plot(MVN_state.t/ms,MVN_state.eyeMovement[0])
	xlabel('time')
	ylabel('Eye movement')
	
show()

import numpy as np
from scipy.stats import norm
np.random.seed(1)
norm.ppf(np.random.rand(1,5))














