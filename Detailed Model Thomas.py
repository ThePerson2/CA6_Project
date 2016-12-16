from brian2 import *
import math
import numpy as np

numberGC = 100
gT = -1        # target gain. Should vary a bit depending on day of training.
pT = 0        # target phase shift.
M0=0.25
M1=0.25
G0=1
G1=1
alpha=0.19
alphavm = 0.0000056 # ms -1 
pie=math.pi
sigma=0.02
errorDelay = 0*ms
TauPG = 900*second
T = 1.666*(second)   # rate at which the platform rotates.

Wig = 2.5
Wigini = 2.5
I0 = Wig*G0 - 0.85
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
	'''G =cos((((2*pie)/T)*t)-(pie/2)-x) + G0 : 1
	x : 1
	''')
# Some phase delays of the above function
	
for i in range(0, numberGC):
	GC.x[i] = (i/numberGC)*pie*2 + alpha*cos((i/numberGC)*pie*2)
# makes it so that the delays are distributed evenly


IC = NeuronGroup(1,
	'''
	I : 1
	Iini : 1
	Gtot : 1
	''')
						
PC = NeuronGroup(1,
	'''
	P1 : 1
	P2 : 1
	P1ini : 1
	P2ini : 1
	P=P1-P2 : 1
	Pini=P1ini-P2ini : 1
	C : 1
	V : 1
	''')  
	
# The functions are defined by the synapses later. P is the purkinje cell activity, V is MVN activity, which is sent back so that it cen be used for the weight calculation.

MVN = NeuronGroup(1, 
	''' 
	P : 1
	Pini : 1
	A : 1
	Ve0 = 2.25 : 1
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
	M0=0.25 : 1
	H=0.03 : 1
	L=1 : 1
	''')

VT=NeuronGroup(1,
	'''
	Vt=abs(gT)*M1*cos((2*pie*t/T)-(pie/2))+Vt0 : 1
	Vt0=1 : 1
	''')


## make the synapses

# Spg = Synapses(GC,PC,model='''P_post = Wpg*G_pre : 1 (summed)
# 								Wpg = x_pre : 1''')

Sig=Synapses(GC,IC,
	'''
	Gtot_post=G_pre : 1 (summed)
	I_post=(Wig/numberGC)*Gtot_post-I0 : 1
	Iini_post=(Wigini/numberGC)*Gtot_post-I0 : 1
	Wig = 2.5 : 1
	Wigini = 2.5 : 1
	I0 = Wig*G0 - 0.85 : 1
	''')
#need to figure out how (summed) works with constants !!!

Sip=Synapses(IC,PC,
	'''
	P2_post=Wpi*I_pre : 1
	P2ini_post=Wpi*Iini_pre : 1
	Wpi=1 : 1
	''')

Spg = Synapses(GC,PC,
	'''
	Wpgini = 1.85  : 1
	alphaD=4.5*(1e-6)*1e3*(1/second) : 1/second
	alphaPG=3.5*(1e-5)*1e3*(1/second) : 1/second
	P1_post = (1/numberGC)*Wpg*G_pre : 1 (summed)
	P1ini_post = (1/numberGC)*Wpgini*G_pre : 1 (summed)
	dWpg/dt = (-alphaPG*C_post+sqrt(alphaPG)*sigma*(1))*G_pre+alphaD*(Wpgini-Wpg) : 1 (clock-driven)
	depsi/dt=randn() : 1 (clock-driven)
	''')
Spg.connect()
Spg.Wpg=1.85
# sums granule cell activity into the purkinje cell, as weights are applied.

Smv = Synapses(MF,MVN,
	'''
	VI_post=M_pre : 1
	A_post=2*Wvm*(M_pre-M0) : 1
	dWvm/dt=alphavm*(M0-M_pre)*(P_post-Pini_post) : 1 (clock-driven)
	alphaVM=5.5*(1e-6)*1e3*(1/second) : 1/second
	''')
Smv.connect()
Smv.Wvm=0.88

Spv = Synapses(PC,MVN,
	'''
	P_post = P_pre : 1
	Pini_post=Pini_pre : 1
	''')

Svc=Synapses(MVN,CF,
	'''
	V_post=V_pre : 1
	''')

Smc=Synapses(MF,CF,
	'''
	M_post=M_pre : 1
	''')
#Smc.delay=errorDelay
	
Stc=Synapses(VT,CF,
	'''
	Vt_post=Vt_pre : 1
	''')
#Stc.delay=errorDelay
	
Scp=Synapses(CF,PC,
	'''
	C_post=C_pre : 1
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
MF_state = StateMonitor(MF,'M',record=0)
GC_state = StateMonitor(GC,'G',record=True)
Weight_state = StateMonitor(Spg,'Wpg',record=True)
PC_state = StateMonitor(PC,'P',record=0)
MVN_state = StateMonitor(MVN,'eyeMovement',record=0)

## run the model

# They are set up like this to immitate the paradigm for the paper. 
defaultclock.dt=0.100*second
run(100*second)

gT = -0.5

#run(5.0*second)

gT = -1

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
	for i in range(0, 5):
		figure()
		subplot(211)
		plot(GC_state.t/ms,GC_state.G[i])
		xlabel('time')
		ylabel('GC')
if 'Weight_state' in locals():
	for i in range(0, numberGC):
		figure()
		# subplot(211)
		plot(Weight_state.t/ms,Weight_state.Wpg[i])
if 'PC_state' in locals():
	figure()
	subplot(211)
	plot(PC_state.t/ms,PC_state.P[0])
	xlabel('time')
	ylabel('PC')
	
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














