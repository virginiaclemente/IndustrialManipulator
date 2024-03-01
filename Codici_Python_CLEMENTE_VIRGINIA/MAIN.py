#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A,  OUTPUT_B,  OUTPUT_C,  OUTPUT_D #, SpeedPercent
from ev3dev2.power import PowerSupply
import numpy as np
from time import sleep

m1 = LargeMotor(OUTPUT_A)
m2 = LargeMotor(OUTPUT_B)
m3 = LargeMotor(OUTPUT_C)
m4 = MediumMotor(OUTPUT_D)

m1.reset
m2.reset
m3.reset
m4.reset

a= PowerSupply() 
Qd = np.loadtxt('Qd_funzionale.txt')
print(len(Qd)) #1001

angles = np.zeros((len(Qd),4))
length = len(Qd)

for i in range(0, length, 1):
    Qd[i, 0] = np.degrees(Qd[i, 0])
    Qd[i, 1] = np.degrees(Qd[i, 1])
    Qd[i, 2] = np.degrees(Qd[i, 2])
    Qd[i, 3] = np.degrees(Qd[i, 3])

# inizializzazione posizione dei motori
# porto il manipolatore nella posizione iniziale desiderata, posso togliere questa parte?
# il manipolatore é posizionato da noi nella posizione iniziale 
#m1.on_to_position(5, 30, brake=True, block=True) 
#m2.on_to_position(5, -30, brake=True, block=True)
#m3.on_to_position(5, 0, brake=True, block=True)
#m4.on_to_position(5, 30,brake=True, block=True)


for i in range(0, length, 1):
    m1.on_to_position(5, -Qd[i,0], brake=True, block=True) # Gira fino all'angolo desiderato con velocità assegnata
    angles[i,0] = m1.position

    m2.on_to_position(5, -Qd[i,1],brake=True, block=True)
    angles[i,1] = m2.position

    m3.on_to_position(5, -Qd[i,2],brake=True, block=True)
    angles[i,2] = m3.position 

    m4.on_to_position(5, -Qd[i,3],brake=True, block=True)
    angles[i,3] = m4.position
    #sleep(.1) 

np.savetxt('angoli.txt', angles)